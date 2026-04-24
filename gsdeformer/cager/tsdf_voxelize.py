import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import open3d as o3d
import open3d.core as o3c
import torch
import tqdm

import gsdeformer.hack_add_gs3d_to_sys_path
from gs3d.arguments import PipelineParams
from gs3d.depth_gaussian_renderer import render
from gs3d.scene.dataset_readers import CameraInfo
from gs3d.scene.gaussian_model import GaussianModel
from gs3d.utils.graphics_utils import focal2fov
from gsdeformer.editorv2.utils_camera import cvt_camera_info_to_camera, cvt_camera_3dgs_to_o3d_intrin, \
    cvt_camera_3dgs_to_o3d_extrin
from gsdeformer.editorv2.utils_data import load_model


@dataclass
class CagingCLIArguments:
    # model options
    model_path: Path = None  # path to trained GS3D folder
    iteration: int = -1  # the iter of model to load, -1 means to pick the latest
    # camera options
    fx: float = None
    fy: float = None
    width: int = None
    height: int = None
    num_rings: int = 5
    num_camera_per_ring: int = 8
    bsphere_opacity_threshold: float = 0.01
    bsphere_expand: float = 2.0
    # T-SDF options
    enable_tsdf: bool = True
    tsdf_resolution: int = 256
    device: str = "CUDA:0"
    # to voxel options
    resolution: int = 256
    bbox_max: str = "1.0,1.0,1.0"
    bbox_min: str = "-1.0,-1.0,-1.0"
    # output options
    expname: str = "" # name of the experiment for output image
    output_path: Path = None # path to place output
    # debug switch
    debug: bool = False


def parse_args(args) -> CagingCLIArguments:
    parser = argparse.ArgumentParser(description='T-SDF Voxelization Tool')

    # model options
    parser.add_argument('--model_path', type=Path, required=True,
                        help='Path to trained GS3D folder')
    parser.add_argument('--iteration', type=int, default=-1,
                        help='Model iteration to load, -1 means latest')

    # camera options
    parser.add_argument('--fx', type=float, required=True,
                        help='Focal length x')
    parser.add_argument('--fy', type=float, required=True,
                        help='Focal length y')
    parser.add_argument('--width', type=int, required=True,
                        help='Image width')
    parser.add_argument('--height', type=int, required=True,
                        help='Image height')
    parser.add_argument('--num_rings', type=int, default=5,
                        help='Number of camera rings')
    parser.add_argument('--num_camera_per_ring', type=int, default=8,
                        help='Number of cameras per ring')
    parser.add_argument('--bsphere_opacity_threshold', type=float, default=0.01,
                        help='Bounding sphere opacity threshold')
    parser.add_argument('--bsphere_expand', type=float, default=2.0,
                        help='Bounding sphere expansion factor')

    # T-SDF options
    parser.add_argument('--disable_tsdf', action='store_true', default=False,
                            help='Disable T-SDF computation')
    parser.add_argument('--tsdf_resolution', type=int, default=256,
                        help='T-SDF resolution')
    parser.add_argument('--device', type=str, default='CUDA:0',
                        help='Device to use (e.g. CUDA:0)')

    # voxel options
    parser.add_argument('--resolution', type=int, default=256,
                        help='Voxel resolution')
    parser.add_argument('--bbox_max', type=str, default=None,
                        help='Maximum bbox coordinates (x,y,z)')
    parser.add_argument('--bbox_min', type=str, default=None,
                        help='Minimum bbox coordinates (x,y,z)')

    # output options
    parser.add_argument('--expname', type=str, default='',
                        help='Experiment name for output')
    parser.add_argument('--output_path', type=Path, required=True,
                        help='Output directory path')

    # debug
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')

    args = parser.parse_args(args)

    return CagingCLIArguments(
        model_path=args.model_path,
        iteration=args.iteration,
        fx=args.fx,
        fy=args.fy,
        width=args.width,
        height=args.height,
        num_rings=args.num_rings,
        num_camera_per_ring=args.num_camera_per_ring,
        bsphere_opacity_threshold=args.bsphere_opacity_threshold,
        bsphere_expand=args.bsphere_expand,
        enable_tsdf=not args.disable_tsdf,
        tsdf_resolution=args.tsdf_resolution,
        device=args.device,
        resolution=args.resolution,
        bbox_max=args.bbox_max,
        bbox_min=args.bbox_min,
        expname=args.expname,
        output_path=args.output_path,
        debug=args.debug
    )

def load_synthetic_cameras(
        config: CagingCLIArguments,
        model: GaussianModel, debug_dump_folder: Optional[Path] = None
) -> List[CameraInfo]:
    # Compute center and radius of bounding sphere
    coords = model.get_xyz
    filtered_coords = coords[(model.get_opacity > config.bsphere_opacity_threshold)[:,0]]
    center = torch.mean(filtered_coords, dim=0)
    radius = torch.max(torch.norm(filtered_coords - center, dim=1))

    # Expand radius
    radius = radius * config.bsphere_expand

    # Generate camera positions on sphere surface
    phi = torch.arange(0, config.num_camera_per_ring, device=center.device).float() * (2 * torch.pi / config.num_camera_per_ring)
    theta = torch.arange(1, config.num_rings+1, device=center.device).float() * (torch.pi / (config.num_rings+2))
    # meshgrid & flatten to get all combinations of (theta,pi)
    theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing='ij')
    theta = theta_grid.reshape(-1)
    phi = phi_grid.reshape(-1)
    # Add top & bottom camera
    theta = torch.cat([theta, torch.zeros(1, device=center.device), torch.ones(1, device=center.device) * torch.pi])
    phi = torch.cat([phi, torch.zeros(1, device=center.device), torch.zeros(1, device=center.device)])

    # Convert spherical to cartesian coordinates
    x = radius * torch.sin(theta) * torch.cos(phi)
    y = radius * torch.sin(theta) * torch.sin(phi)
    z = radius * torch.cos(theta)
    positions = torch.stack([x, y, z], dim=1) + center

    # Create camera info objects
    cameras = []
    for pos in positions:
        R = look_at_rotation(pos.unsqueeze(0), center.unsqueeze(0))[0,...]
        R[:3, 1:3] *= -1 # OpenGL convention to COLMAP convention

        # Convert W2C to C2W
        c2w = np.eye(4)
        c2w[:3, :3] = R.cpu().numpy()
        c2w[:3, 3] = pos.cpu().numpy()
        w2c = np.linalg.inv(c2w)
        R = w2c[:3, :3]
        T = w2c[:3, 3]

        fovy = focal2fov(config.fy, config.width)
        fovx = focal2fov(config.fx, config.height)
        cam = CameraInfo(
            uid=0,
            R=R.T,
            T=T,
            FovY=fovy,
            FovX=fovx,
            image=None,
            image_path="",
            image_name="",
            width=config.width,
            height=config.height
        )
        cameras.append(cam)

    if debug_dump_folder:
        debug_dump_folder.mkdir(parents=True, exist_ok=True)
        torch.save(cameras, debug_dump_folder / "dbg_synth_cameras.pt")
        torch.save((radius, center), debug_dump_folder / "dbg_synth_sphere.pt")
        torch.save(coords, debug_dump_folder / "dbg_synth_coords.pt")

    return cameras

def normalize_depth(depth: torch.Tensor) -> torch.Tensor:
    non_zero = depth[depth > 0]

    min_val = non_zero.min()
    max_val = torch.quantile(non_zero, 0.99)

    depth[depth == 0.0] = max_val
    depth = torch.clamp((depth - min_val) / (max_val - min_val), 0, 1)
    depth = 1 - depth

    return depth

def look_at_rotation(eye: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    """Computes rotation matrix to look from eye position toward center position.

    Args:
        eye: Camera positions of shape (B, 3)
        center: Look-at points of shape (B, 3)

    Returns:
        Rotation matrices of shape (B, 3, 3)
    """
    batch_size = eye.shape[0]

    # Compute forward direction (z-axis)
    forward = center - eye
    forward = forward / torch.norm(forward, dim=1, keepdim=True)

    # Compute right direction (x-axis)
    # Using world up vector (0,1,0) as reference
    up = torch.tensor([0.0, 1.0, 0.0], device=eye.device).expand(batch_size, 3)
    right = torch.cross(forward, up, dim=1)
    right = right / torch.norm(right, dim=1, keepdim=True)

    # Compute camera up direction (y-axis)
    cam_up = torch.cross(right, forward, dim=1)

    # Stack to create rotation matrix
    rotation = torch.stack([right, cam_up, -forward], dim=2)  # Note: negative forward for OpenGL convention

    return rotation

def _run_tsdf_integration(cameras, depths, voxel_size, device):
    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=('tsdf', 'weight'),
        attr_dtypes=(o3c.float32, o3c.float32),
        attr_channels=((1), (1)),
        voxel_size=voxel_size,
        block_resolution=16,
        block_count=50000,
        device=device
    )
    depth_scale = 1.0
    depth_max = cvt_camera_info_to_camera(cameras[0]).zfar
    vbg_block_coords = []
    for camera, depth in zip(tqdm.tqdm(cameras), depths):
        intrin, extrin = cvt_camera_3dgs_to_o3d_intrin(camera), cvt_camera_3dgs_to_o3d_extrin(camera)
        intrin = o3d.core.Tensor(np.array(intrin.intrinsic_matrix).astype(np.float64))
        extrin = o3d.core.Tensor(np.array(extrin).astype(np.float64))

        depth = o3d.t.geometry.Image(depth.numpy()).to(device)

        frustum_block_coords = vbg.compute_unique_block_coordinates(
            depth, intrin, extrin,
            depth_scale=depth_scale,
            depth_max=depth_max
        )
        vbg.integrate(
            block_coords=frustum_block_coords,
            depth=depth, intrinsic=intrin, extrinsic=extrin,
            depth_scale=depth_scale,
            depth_max=depth_max
        )
        vbg_block_coords.append(frustum_block_coords)
    return vbg, vbg_block_coords


def _render_images(args, model, cameras):
    images = []
    depths = []
    pipe = PipelineParams(argparse.ArgumentParser())  # mock pipeline params
    bg_color = torch.zeros((3,), dtype=torch.float32, device="cuda")
    for idx, camera in enumerate(tqdm.tqdm(cameras)):
        camera = cvt_camera_info_to_camera(camera)
        rendered = render(camera, model, pipe, bg_color)
        image = torch.clamp(rendered["render"], 0.0, 1.0)
        depth = rendered["depth_3dgs"][0].cpu()
        images.append(image)
        depths.append(depth)
    return depths, images


def _load_cameras(args, model):
    cameras = load_synthetic_cameras(
        args, model,
        debug_dump_folder=(args.output_path if args.debug else None)
    )
    return cameras

def _parse_bbox_literal(s: str) -> np.ndarray:
    elem = s.split(",")
    assert len(elem) == 3
    elem = [float(e) for e in elem]
    return np.array(elem)

@torch.no_grad()
def main(args=None):
    args = parse_args(args)
    print("running wih configuration: \n{}", args)

    print("loading model & cameras")
    model = load_model(args.model_path, iteration=args.iteration)
    cameras = _load_cameras(args, model)
    xyz = model.get_xyz

    print("rendering")
    depths, images = _render_images(args, model, cameras)

    if args.enable_tsdf:
        print("computing T-SDF integration")
        xyz_bound_hi, xyz_bound_lo = xyz.max(dim=0)[0], xyz.min(dim=0)[0]
        bbox_max_lens = max((xyz_bound_hi - xyz_bound_lo).tolist())
        resolution = args.tsdf_resolution
        device = o3d.core.Device(args.device)
        voxel_size = bbox_max_lens / resolution
        vbg, vbg_block_coords = _run_tsdf_integration(cameras, depths, voxel_size, device)
    else:
        vbg, vbg_block_coords = None, None

    # calculate voxel grid statistics
    if args.bbox_min or args.bbox_max:
        bbox_min = _parse_bbox_literal(args.bbox_min)
        bbox_max = _parse_bbox_literal(args.bbox_max)
    else:
        bbox_min = xyz.min(dim=0)[0].cpu().numpy()
        bbox_max = xyz.max(dim=0)[0].cpu().numpy()
    bbox_max_lens = max((bbox_max - bbox_min).tolist())
    voxel_size = bbox_max_lens / args.resolution

    print("rendering carving depth map")

    carving_depth = "direct"

    pts = vbg.extract_point_cloud()

    voxel_render = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
        pts.to_legacy(), voxel_size,
        min_bound=bbox_min.tolist(),
        max_bound=bbox_max.tolist()
    )

    og_depths = depths
    if carving_depth == "points":
        assert vbg is not None
        depths = _render_depth_images(cameras, voxel_render)
    elif carving_depth == "direct":
        assert vbg is not None
        depths = _render_depth_images_vbg(cameras, depths, vbg, vbg_block_coords)
    elif carving_depth == "3dgs":
        depths = [d.cpu().numpy() for d in depths]
    else:
        assert False

    print("perform space carving")
    origin = bbox_min
    vret = _carve_voxels(cameras, depths, origin, args.resolution, voxel_size)
    vret = vret + voxel_render

    print("packing voxels")
    # to pack dense voxel, simplified from filter_density_to_packed_grid
    pack_res = args.resolution // 2
    pack_vox_grid = np.zeros(pack_res ** 3, dtype=np.uint8)

    for voxel in vret.get_voxels():
        pos = voxel.grid_index
        pack_position = pos // 2
        num_pack_shift = (pos[0] % 2 + 2 * (pos[1] % 2) + 4 * (pos[2] % 2))

        pack_val = 1 << num_pack_shift
        pack_idx = (pack_position[0] +
                    pack_res * pack_position[1] +
                    pack_res * pack_res * pack_position[2])

        pack_vox_grid[pack_idx] |= pack_val

    print("saving")
    args.output_path.mkdir(parents=True, exist_ok=True)
    prefix = args.expname if args.expname else args.model_path.name
    np.save(str(args.output_path / f"{prefix}_packed_voxels.npy"), pack_vox_grid)
    if args.debug:
        if pts is not None:
            o3d.t.io.write_point_cloud(str(args.output_path / f"{prefix}_pts.pcd"), pts)

        pc_ret = o3d.geometry.PointCloud()
        pc_ret_pts = np.asarray([vret.origin + pt.grid_index * vret.voxel_size for pt in vret.get_voxels()])
        pc_ret.points = o3d.utility.Vector3dVector(pc_ret_pts)

        o3d.io.write_point_cloud(str(args.output_path / f"{prefix}_carved.pcd"), pc_ret)

def _render_depth_images_vbg(cameras, og_depths, vbg, block_coords):
    # carve voxel grid
    depths = []
    for idx, (camera, depth, block_coord) in tqdm.tqdm(list(enumerate(zip(cameras ,og_depths, block_coords))), desc="Rendering"):
        intrinsic = cvt_camera_3dgs_to_o3d_intrin(camera)
        intrinsic = np.ascontiguousarray(np.asarray(intrinsic.intrinsic_matrix))
        extrinsic = cvt_camera_3dgs_to_o3d_extrin(camera)

        depth_min = depth.min().cpu().item()
        depth_max = depth.max().cpu().item()

        result = vbg.ray_cast(
            block_coords=block_coord,
            intrinsic=intrinsic,
            extrinsic=extrinsic,
            width=camera.width,
            height=camera.height,
            render_attributes=['depth'],
            depth_scale=1.0,
            depth_min=depth_min,
            depth_max=depth_max,
            trunc_voxel_multiplier=1.0
        )
        depth = result['depth']
        depth = depth.cpu().numpy()[...,0]

        depths.append(depth)

    return depths

def _render_depth_images(cameras, voxel_render):
    # setup visualizer to render depth maps
    vis = o3d.visualization.Visualizer()
    ref_cam = cameras[0]
    vis.create_window(width=ref_cam.width, height=ref_cam.height, visible=False)
    vis.add_geometry(voxel_render)
    ctr = vis.get_view_control()
    # carve voxel grid
    depths = []
    for camera in tqdm.tqdm(cameras, desc="Carving"):
        param = o3d.camera.PinholeCameraParameters()
        param.intrinsic = cvt_camera_3dgs_to_o3d_intrin(camera)
        param.extrinsic = cvt_camera_3dgs_to_o3d_extrin(camera)
        ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)

        # capture depth image and make a point cloud
        vis.poll_events()
        vis.update_renderer()
        depth = vis.capture_depth_float_buffer(False)

        depths.append(depth)
    vis.destroy_window()
    return depths


def _carve_voxels(cameras, depths, origin, resolution, voxel_size):
    # setup grid for carving
    voxel_carving = o3d.geometry.VoxelGrid.create_dense(
        width=resolution * voxel_size,
        height=resolution * voxel_size,
        depth=resolution * voxel_size,
        voxel_size=voxel_size,
        origin=origin,
        color=[0.0, 0.0, 0.0]
    )
    # carve voxel grid
    for camera, depth in tqdm.tqdm(list(zip(cameras, depths)), desc="Carving"):
        param = o3d.camera.PinholeCameraParameters()
        param.intrinsic = cvt_camera_3dgs_to_o3d_intrin(camera)
        param.extrinsic = cvt_camera_3dgs_to_o3d_extrin(camera)

        # depth map carving method
        voxel_carving.carve_depth_map(o3d.geometry.Image(depth), param)
        # voxel_carving.carve_silhouette(o3d.geometry.Image(depth), param)
    vret = voxel_carving
    return vret


if __name__ == '__main__':
    main()