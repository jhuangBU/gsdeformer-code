import json
import os
from pathlib import Path

import dill
import numpy as np
import torch
import trimesh

from algorithm import deform_mesh
from frosting_scene.cameras import GSCamera, CamerasWrapper
from frosting_scene.gs_model import GaussianSplattingWrapper
from frosting_scene.frosting_model import load_frosting_model
from blender.frosting_utils import build_composite_scene
from torchvision.utils import save_image
import argparse
import open3d as o3d

def load_refined_mesh(refined_sugar_path):
    checkpoint = torch.load(refined_sugar_path)
    with torch.no_grad():
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(checkpoint['state_dict']['_shell_base_verts'].cpu().numpy())
        o3d_mesh.triangles = o3d.utility.Vector3iVector(checkpoint['state_dict']['_shell_base_faces'].cpu().numpy())
    return o3d_mesh

def main():
    warmup_iter = 5
    benchmark_iter = 10

    parser = argparse.ArgumentParser(description="Render images using Frosting model")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--source_path", type=Path, required=True)
    parser.add_argument("--gs_folder", type=Path, required=True)
    parser.add_argument("--frosting_checkpoint", type=Path, required=True)
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument("--cam_idx", type=int, default=34)
    parser.add_argument("--override_cam", type=Path, default=None)
    parser.add_argument(
        "--use_occlusion_culling", action="store_true",
        help="If your model was trained with occlusion culling, set use_occlusion_culling to True"
    )
    parser.add_argument("--src_cage", type=Path, default=None)
    parser.add_argument("--dst_cage", type=Path, default=None)
    parser.add_argument("--output_folder", type=Path, required=True)
    parser.add_argument("--benchmark", action="store_true")

    parser.add_argument("--use_complex_adapt", action="store_true",
                        help="Use complex adaptation method")
    parser.add_argument("--thickness_rescaling_method", type=str, default='median',
                        choices=['median', 'triangle'],
                        help="Method for rescaling thickness (default: median)")
    parser.add_argument("--deformation_threshold", type=float, default=2.0,
                        help="Threshold for deformation (default: 2.0), 0.0 for none")

    args = parser.parse_args()

    torch.cuda.set_device(f'cuda:{args.gpu}')
    device = torch.device(torch.cuda.current_device())

    source_path = str(args.source_path)
    if source_path[-1] != '/':
        source_path += '/'
    gs_output_dir = f'{args.gs_folder}/'
    frosting_path = str(args.frosting_checkpoint)

    print("Path to the COLMAP dataset: ", source_path, sep='\n')
    print("Path to the vanilla 3DGS output directory: ", gs_output_dir, sep='\n')
    print("Path to the Frosting model: ", frosting_path, sep='\n')

    eval_split = False
    white_background = args.white_background
    bg_color = [1., 1., 1.] if white_background else [0., 0., 0.]

    # Load corresponding Vanilla 3DGS Model
    gs_model = GaussianSplattingWrapper(
        source_path=source_path,
        output_path=gs_output_dir if gs_output_dir.endswith(os.sep) else gs_output_dir + os.sep,
        iteration_to_load=7000,
        load_gt_images=False,
        eval_split=eval_split,
        eval_split_interval=8,
        background=bg_color,
        white_background=white_background,
        remove_camera_indices=[],
    )

    # Load Frosting model
    lfrosting = load_frosting_model(frosting_path, nerfmodel=gs_model)
    o3d_mesh = load_refined_mesh(frosting_path)
    if args.src_cage and args.dst_cage:

        event_preproc = torch.cuda.Event(enable_timing=True)
        event_postproc = torch.cuda.Event(enable_timing=True)
        event_postdeform = torch.cuda.Event(enable_timing=True)
        preprocess_times = []
        deform_times = []

        src_cage, dst_cage = trimesh.load(args.src_cage), trimesh.load(args.dst_cage)
        for _ in range(benchmark_iter if args.benchmark else 1):
            event_preproc.record()
            deformer = deform_mesh(o3d_mesh, src_cage)
            event_postproc.record()
            do3d_mesh = deformer(dst_cage)
            frosting = build_composite_scene(
                {'dummy': lfrosting},
                ['dummy'],
                {
                    'bones': [None],
                    'meshes': [{
                    'checkpoint_name': "dummy",
                    'xyz': torch.tensor(np.asarray(do3d_mesh.vertices), dtype=torch.float),
                    'matrix_world': torch.eye(4),
                    "idx": slice(None)  # keep everything as-is
                }]},
                # default values of render_blender_scene.py
                use_simple_adapt=not args.use_complex_adapt,
                thickness_rescaling_method=args.thickness_rescaling_method,
                deformation_threshold=(args.deformation_threshold if args.deformation_threshold != 0.0 else None),
            )
            event_postdeform.record()
            torch.cuda.synchronize()
            preprocess_times.append(event_preproc.elapsed_time(event_postproc))
            deform_times.append(event_postproc.elapsed_time(event_postdeform))
    else:
        print("RENDERING ORIGINAL, NO DEFORMATION APPLIED")
        frosting = lfrosting

    print(f"Number of vertices in base mesh: {len(frosting._shell_base_verts)}")
    print(f"Number of faces in base mesh: {len(frosting._shell_base_faces)}")
    print(f"Number of Gaussians in the frosting layer: {len(frosting._bary_coords)}")
    if frosting.use_background_gaussians:
        print(f"Number of Gaussians in the background: {len(frosting._bg_points)}")

    # Render
    use_occlusion_culling = args.use_occlusion_culling
    override_cam = args.override_cam
    cam_idx = args.cam_idx

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
            # Batch mode: render all cameras for quantitative evaluation
            cameras_to_use = gs_model.training_cameras
            cam_indices_to_render = list(range(len(cameras_to_use.gs_cameras)))
            print(f"Rendering all {len(cam_indices_to_render)} cameras for quantitative evaluation")
        else:
            # Single camera (existing behavior)
            cameras_to_use = None
            cam_indices_to_render = [cam_idx]
            print(f"Rendering image with index {cam_idx}.")

    # Unified rendering loop
    args.output_folder.mkdir(parents=True, exist_ok=True)
    render_times = []

    with torch.no_grad():
        for idx in cam_indices_to_render:
            if cameras_to_use:
                camera = cameras_to_use.gs_cameras[idx]
                frosting.image_height = cameras_to_use.height[idx].cpu().item()
                frosting.image_width = cameras_to_use.width[idx].cpu().item()

            if len(cam_indices_to_render) > 1:
                print(f"  Rendering camera {idx+1}/{len(cam_indices_to_render)}: {camera.image_name}")

            event_prerender = torch.cuda.Event(enable_timing=True)
            event_postrender = torch.cuda.Event(enable_timing=True)
            for _ in range(benchmark_iter if args.benchmark else 1):
                event_prerender.record()
                rgb_img = frosting.render_image_gaussian_rasterizer(
                    nerf_cameras=cameras_to_use,
                    camera_indices=idx,
                    compute_color_in_rasterizer=True,
                    use_occlusion_culling=use_occlusion_culling,
                    bg_color=torch.tensor(bg_color, device=device),
                ).clamp(0, 1)
                event_postrender.record()
                torch.cuda.synchronize()
                render_times.append(event_prerender.elapsed_time(event_postrender))

            # Determine filename based on mode
            if len(cam_indices_to_render) == 1:
                # Single camera: preserve existing naming
                suffix = "" if args.src_cage else "_og"
                filename = f"rendered{suffix}.png"
            else:
                # Batch mode: use camera's image_name for exact matching with GT
                filename = f"{camera.image_name}.png"

            save_image(rgb_img.permute(2, 0, 1), args.output_folder / filename)

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
        (args.output_folder / "benchmark.json").write_text(json.dumps(benchmark_data, indent=4), encoding="UTF-8")

if __name__ == "__main__":
    main()