import argparse
import json
import os
from pathlib import Path
from typing import Optional

import dill
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import open3d.geometry as ogeo
import open3d.utility as outil
import torch
import torchvision.utils
import trimesh
from scipy.spatial import Delaunay

from algorithm import euclidean_to_mvc, deform_mvc_to_euclidean
from blender.sugar_utils import build_composite_scene
from sugar_scene.cameras import CamerasWrapper, GSCamera
from sugar_scene.gs_model import GaussianSplattingWrapper
from sugar_scene.sugar_model import SuGaR
from sugar_utils.spherical_harmonics import SH2RGB

def deform_mesh(mesh: ogeo.TriangleMesh, src_cage: trimesh.Trimesh):
    # calculate deforming mask
    cage_vertices = ogeo.PointCloud(outil.Vector3dVector(src_cage.vertices))
    cage_hull, _ = cage_vertices.compute_convex_hull()
    hull = Delaunay(np.asarray(cage_hull.vertices))
    mesh_vertices = np.asarray(mesh.vertices).copy()
    mesh_vertices_mask = hull.find_simplex(mesh_vertices) >= 0

    deform_vertices = mesh_vertices[mesh_vertices_mask]
    deform_vertices = torch.tensor(deform_vertices).cuda()
    mean_vals = euclidean_to_mvc(src_cage, deform_vertices)

    def _deform(dst_cage: trimesh.Trimesh) -> ogeo.TriangleMesh:
        deformed_vertices, _ = deform_mvc_to_euclidean(dst_cage, mean_vals, deform_vertices)

        ret_vertices = mesh_vertices.copy()
        ret_vertices[mesh_vertices_mask] = deformed_vertices.cpu().numpy()
        ret_vertices = ret_vertices.astype(np.float64)
        ret_mesh = ogeo.TriangleMesh(
            vertices=outil.Vector3dVector(ret_vertices),
            triangles=outil.Vector3iVector(np.asarray(mesh.triangles))
        )
        ret_mesh.vertex_colors=outil.Vector3dVector(np.asarray(mesh.vertex_colors))

        return ret_mesh

    return _deform

def load_refined_mesh(refined_sugar_path):
    checkpoint = torch.load(refined_sugar_path)

    with torch.no_grad():
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(checkpoint['state_dict']['_points'].cpu().numpy())
        o3d_mesh.triangles = o3d.utility.Vector3iVector(checkpoint['state_dict']['_surface_mesh_faces'].cpu().numpy())
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(
            torch.ones_like(checkpoint['state_dict']['_points']).cpu().numpy())

    return o3d_mesh


def load_refined_model(refined_sugar_path, nerfmodel: GaussianSplattingWrapper):
    checkpoint = torch.load(refined_sugar_path, map_location=nerfmodel.device)
    n_faces = checkpoint['state_dict']['_surface_mesh_faces'].shape[0]
    n_gaussians = checkpoint['state_dict']['_scales'].shape[0]
    n_gaussians_per_surface_triangle = n_gaussians // n_faces

    print("Loading refined model...")
    print(f'{n_faces} faces detected.')
    print(f'{n_gaussians} gaussians detected.')
    print(f'{n_gaussians_per_surface_triangle} gaussians per surface triangle detected.')

    with torch.no_grad():
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(checkpoint['state_dict']['_points'].cpu().numpy())
        o3d_mesh.triangles = o3d.utility.Vector3iVector(checkpoint['state_dict']['_surface_mesh_faces'].cpu().numpy())
        # o3d_mesh.vertex_normals = o3d.utility.Vector3dVector(normals.cpu().numpy())
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(
            torch.ones_like(checkpoint['state_dict']['_points']).cpu().numpy())

    refined_sugar = SuGaR(
        nerfmodel=nerfmodel,
        points=checkpoint['state_dict']['_points'],
        colors=SH2RGB(checkpoint['state_dict']['_sh_coordinates_dc'][:, 0, :]),
        initialize=False,
        sh_levels=nerfmodel.gaussians.active_sh_degree + 1,
        keep_track_of_knn=False,
        knn_to_track=0,
        beta_mode='average',
        surface_mesh_to_bind=o3d_mesh,
        n_gaussians_per_surface_triangle=n_gaussians_per_surface_triangle,
    )
    refined_sugar.load_state_dict(checkpoint['state_dict'])

    return refined_sugar


def parse_arguments(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_cage", type=Path, default=None)
    parser.add_argument("--dst_cage", type=Path, default=None)
    parser.add_argument("--source_path", type=Path)
    parser.add_argument("--gs_folder", type=Path)
    parser.add_argument("--gs_iter", type=int, default=7000)
    parser.add_argument("--refined_sugar_folder", type=Path)
    parser.add_argument("--refined_sugar_iter", type=int, default=15_000)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cam_idx", type=int, default=0)
    parser.add_argument("--override_cam", type=Path, default=None)
    parser.add_argument("--output_folder", type=Path)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--white_bg", action="store_true")
    return parser.parse_args(args)

@torch.no_grad()
def main(args=None):
    warmup_iter = 5
    benchmark_iter = 10

    args = parse_arguments(args)
    src_cage: Path = args.src_cage
    dst_cage: Path = args.dst_cage
    source_path: str = str(args.source_path)
    gs_folder: str = str(args.gs_folder) + os.sep
    gs_iter: int = args.gs_iter
    refined_sugar_folder: str = str(args.refined_sugar_folder) + os.sep
    refined_sugar_iter: int = args.refined_sugar_iter
    gpu: int = args.gpu
    cam_idx: int = args.cam_idx
    override_cam: Optional[Path] = args.override_cam
    output_folder: Path = args.output_folder

    torch.cuda.set_device(gpu)

    # ====================Load NeRF model and training data====================

    # Load Scene & Gaussian Splatting checkpoint
    print(f"\nLoading config {gs_folder}...")
    nerfmodel = GaussianSplattingWrapper(
        source_path=source_path,
        output_path=gs_folder,
        iteration_to_load=gs_iter,
        load_gt_images=False,
        eval_split=False,
        eval_split_interval=8,
    )

    print(f'{len(nerfmodel.training_cameras)} training images detected.')
    print(f'The model has been trained for {gs_iter} steps.')
    print(len(nerfmodel.gaussians._xyz) / 1e6, "M gaussians detected.")

    # Load the refined SuGaR checkpoint.
    event_preproc = torch.cuda.Event(enable_timing=True)
    event_postproc = torch.cuda.Event(enable_timing=True)
    event_postdeform = torch.cuda.Event(enable_timing=True)
    preprocess_times = []
    deform_times = []
    refined_sugar_path = os.path.join(refined_sugar_folder, f"{refined_sugar_iter}.pt")
    print(f"\nLoading SuGaR {refined_sugar_path}...")
    o3d_mesh = load_refined_mesh(refined_sugar_path)
    if src_cage and dst_cage:
        src_cage, dst_cage = trimesh.load(src_cage), trimesh.load(dst_cage)
    loaded_sugar = load_refined_model(refined_sugar_path, nerfmodel)
    if src_cage and dst_cage:
        for _ in range(benchmark_iter if args.benchmark else 1):
            event_preproc.record()
            deformer = deform_mesh(o3d_mesh, src_cage)
            event_postproc.record()
            do3d_mesh = deformer(dst_cage)
            refined_sugar = build_composite_scene(
                {'dummy': loaded_sugar},
                ['dummy'],
                {'meshes':[{
                    'checkpoint_name': "dummy",
                    'xyz': torch.tensor(np.asarray(do3d_mesh.vertices), dtype=torch.float),
                    'matrix_world': torch.eye(4),
                    "idx": slice(None) # keep everything as-is
                }]}
            )
            event_postdeform.record()
            torch.cuda.synchronize()
            preprocess_times.append(event_preproc.elapsed_time(event_postproc))
            deform_times.append(event_postproc.elapsed_time(event_postdeform))
    else:
        refined_sugar = loaded_sugar
        print("SKIPPING DEFORMATION, rendering ORIGINAL IMAGE")

    # ====================Render Images====================

    cameras_to_use = nerfmodel.training_cameras
    cam_idx = args.cam_idx

    refined_sugar.eval()
    refined_sugar.adapt_to_cameras(cameras_to_use)

    # Determine which cameras to render
    if override_cam:
        print("Rendering image with overridden camera")
        cam = torch.load(override_cam, pickle_module=dill)
        cam = GSCamera(
            cam.uid, cam.R, cam.T, cam.FovX, cam.FovY,
            None, None, cam.image_name, cam.uid,
            trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
            image_height=cam.height, image_width=cam.width
        )
        cameras_to_use = CamerasWrapper([cam])
        cam_indices_to_render = [0]
    else:
        if cam_idx == -1:
            # Render all cameras for quantitative evaluation
            cam_indices_to_render = list(range(len(cameras_to_use.gs_cameras)))
            print(f"Rendering all {len(cam_indices_to_render)} cameras for quantitative evaluation")
        else:
            # Single camera (existing behavior)
            cam_indices_to_render = [cam_idx]
            print(f"Rendering image with index {cam_idx}.")
            print("Image name:", cameras_to_use.gs_cameras[cam_idx].image_name)

    # Unified rendering loop
    output_folder.mkdir(parents=True, exist_ok=True)
    render_times = []

    with torch.no_grad():
        for idx in cam_indices_to_render:
            camera = cameras_to_use.gs_cameras[idx]
            if len(cam_indices_to_render) > 1:
                print(f"  Rendering camera {idx+1}/{len(cam_indices_to_render)}: {camera.image_name}")

            def _render():
                refined_sugar.image_height = cameras_to_use.height[idx].cpu().item()
                refined_sugar.image_width = cameras_to_use.width[idx].cpu().item()
                sugar_image = refined_sugar.render_image_gaussian_rasterizer(
                    nerf_cameras=cameras_to_use,
                    camera_indices=idx,
                    bg_color=(1.0 if args.white_bg else 0.0) * torch.Tensor([1.0, 1.0, 1.0]).to(refined_sugar.device),
                    sh_deg=nerfmodel.gaussians.active_sh_degree,
                    compute_color_in_rasterizer=True,
                ).nan_to_num().clamp(min=0, max=1)
                return sugar_image

            event_prerender = torch.cuda.Event(enable_timing=True)
            event_postrender = torch.cuda.Event(enable_timing=True)
            for _ in range(benchmark_iter if args.benchmark else 1):
                event_prerender.record()
                sugar_image = _render()
                event_postrender.record()
                torch.cuda.synchronize()
                render_times.append(event_prerender.elapsed_time(event_postrender))

            # Determine filename based on mode
            if len(cam_indices_to_render) == 1:
                # Single camera: preserve existing naming
                filename = "rendered.png" if src_cage else "rendered_og.png"
            else:
                # Batch mode: use camera's image_name
                filename = f"{camera.image_name}.png"

            torchvision.utils.save_image(sugar_image.permute(2,0,1), output_folder / filename)

        if args.benchmark:
            preprocess_times = preprocess_times[warmup_iter:]
            deform_times = deform_times[warmup_iter:]
            render_times = render_times[warmup_iter:]
            benchmark_data = {
                "time_preproc_ms": preprocess_times,
                "time_deform_ms": deform_times,
                "time_render_ms": render_times,
                "avg_time_preproc_ms": sum(preprocess_times) / len(preprocess_times) if preprocess_times else 0,
                "avg_time_deform_ms": sum(deform_times) / len(deform_times) if deform_times else 0,
                "avg_time_render_ms": sum(render_times) / len(render_times) if render_times else 0
            }
            (output_folder / "benchmark.json").write_text(json.dumps(benchmark_data, indent=4), encoding="UTF-8")

if __name__ == '__main__':
    main()