#
# Copyright (C) 2024, Gmum
# Group of Machine Learning Research. https://gmum.net/
# All rights reserved.
#
# The Gaussian-splatting software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
# For inquiries contact  george.drettakis@inria.fr
#
# The Gaussian-mesh-splatting is software based on Gaussian-splatting, used on research.
# This Games software is free for non-commercial, research and evaluation use
#
import json
from pathlib import Path
from typing import Optional

import dill
import numpy as np
import torch
import trimesh
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from renderer.gaussian_points_animated_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from games.flat_splatting.scene.points_gaussian_model import PointsGaussianModel

from scene.cameras import Camera
from .algorithm import euclidean_to_mvc, deform_mvc_to_euclidean

warmup_iter = 5
benchmark_iter = 10

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, src_cage, dst_cage, benchmark, preprocess_times):
    if src_cage and dst_cage:
        src_cage = trimesh.load(src_cage)
        dst_cage = trimesh.load(dst_cage)
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "cage_deformed_gs_points")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    v1, v2, v3 = gaussians.v1, gaussians.v2, gaussians.v3
    triangles = torch.stack([v1, v2, v3], dim=1)

    pre = torch.cuda.Event(enable_timing=benchmark)
    post_prep = torch.cuda.Event(enable_timing=benchmark)
    post_deform = torch.cuda.Event(enable_timing=benchmark)
    preprocess_times2 = []
    deform_times = []
    if benchmark:
        for _ in range(benchmark_iter):
            vertices = triangles.reshape(-1, 3).cuda()
            pre.record()
            mean_vals = euclidean_to_mvc(src_cage, vertices)
            post_prep.record()
            deformed_vertices, _ = deform_mvc_to_euclidean(dst_cage, mean_vals, vertices)
            new_triangles = deformed_vertices.reshape(-1, 3, 3)
            post_deform.record()
            torch.cuda.synchronize()
            preprocess_times2.append(pre.elapsed_time(post_prep))
            deform_times.append(post_prep.elapsed_time(post_deform))
    else:
        if src_cage and dst_cage:
            vertices = triangles.reshape(-1, 3).cuda()
            mean_vals = euclidean_to_mvc(src_cage, vertices)
            deformed_vertices, _ = deform_mvc_to_euclidean(dst_cage, mean_vals, vertices)
            new_triangles = deformed_vertices.reshape(-1, 3, 3)
            del mean_vals
            del deformed_vertices
            del vertices
        else:
            print("SKIPPING DEFORMATION, rendering ORIGINAL IMAGE")
            new_triangles = triangles

    called = False
    pre_render = torch.cuda.Event(enable_timing=benchmark)
    post_render = torch.cuda.Event(enable_timing=benchmark)
    render_times = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        called = True
        pre_render.record()
        rendering = render(new_triangles, view, gaussians, pipeline, background)["render"]
        post_render.record()
        if benchmark and idx < benchmark_iter:
            torch.cuda.synchronize()
            render_times.append(pre_render.elapsed_time(post_render))
        if benchmark and idx >= benchmark_iter:
            break
        gt = view.original_image[0:3, :, :]
        suffix = "" if src_cage else "_og"
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}{1}'.format(idx, suffix) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

    if benchmark:
        torch.cuda.synchronize()
        def safe_mean(arr): return sum(arr) / len(arr) if arr else 0
        preprocess_times = preprocess_times[warmup_iter:]
        preprocess_times2 = preprocess_times2[warmup_iter:]
        deform_times = deform_times[warmup_iter:]
        render_times = render_times[warmup_iter:]
        d = {
            "preprocess_times_ms": preprocess_times,
            "preprocess_times2_ms": preprocess_times2,
            "deform_times_ms": deform_times,
            "render_times_ms": render_times,
            "avg_preprocess_time_ms": safe_mean(preprocess_times),
            "avg_preprocess_time2_ms": safe_mean(preprocess_times2),
            "avg_deform_time_ms": safe_mean(deform_times),
            "avg_render_time_ms": safe_mean(render_times)
        }
        (Path(render_path) / "benchmark.json").write_text(json.dumps(d, indent=4), encoding="UTF-8")

def render_sets(
        dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool,
        src_cage: str, dst_cage: str, override_cam: Optional[Path], benchmark: bool, store_proxy: bool
):
    with torch.no_grad():
        pre = torch.cuda.Event(enable_timing=benchmark)
        post = torch.cuda.Event(enable_timing=benchmark)
        preprocess_times = []
        if benchmark:
            for _ in range(benchmark_iter):
                gaussians = PointsGaussianModel(dataset.sh_degree)
                scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
                pre.record()
                if hasattr(gaussians, 'prepare_vertices'):
                    gaussians.prepare_vertices()
                if hasattr(gaussians, 'prepare_scaling_rot'):
                    gaussians.prepare_scaling_rot()
                post.record()
                torch.cuda.synchronize()
                preprocess_times.append(pre.elapsed_time(post))
        else:
            gaussians = PointsGaussianModel(dataset.sh_degree)
            scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
            if hasattr(gaussians, 'prepare_vertices'):
                gaussians.prepare_vertices()
            if hasattr(gaussians, 'prepare_scaling_rot'):
                gaussians.prepare_scaling_rot()

        if store_proxy:
            v1, v2, v3 = gaussians.v1, gaussians.v2, gaussians.v3
            triangles = torch.stack([v1, v2, v3], dim=1)
            triangles = triangles.reshape(-1, 3)
            faces = torch.zeros((triangles.shape[0] // 3, 3), device="cpu", dtype=torch.int)
            faces[:, 0] = 0
            faces[:, 1] = 1
            faces[:, 2] = 2
            delta = (torch.arange(0, faces.shape[0], device="cpu", dtype=torch.int) * 3)[:,None]
            faces = faces + delta
            mesh = trimesh.Trimesh(
                vertices=triangles.cpu().numpy(),
                faces=faces.cpu().numpy()
            )
            p = os.path.join(dataset.model_path, "train", "ours_{}".format(iteration))
            makedirs(p, exist_ok=True)
            mesh.export(os.path.join(p, "proxy_mesh.ply"))

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if override_cam:
            cam = torch.load(override_cam, pickle_module=dill)
            image = torch.zeros((3, cam.height, cam.width)).cuda()
            cam = Camera(
                cam.uid, cam.R, cam.T, cam.FovX, cam.FovY,
                image, None, cam.image_name, cam.uid,
                trans = np.array([0.0, 0.0, 0.0]), scale = 1.0, data_device = "cuda"
            )
            train_cameras = [cam]
        else:
            train_cameras = scene.getTrainCameras()

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, train_cameras, gaussians, pipeline, background, src_cage, dst_cage, benchmark, preprocess_times)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, src_cage, dst_cage, benchmark, preprocess_times)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--gs_type', type=str, default="gs_points")
    parser.add_argument("--num_splats", type=int, default=2)
    parser.add_argument("--src_cage", type=str, default=None)
    parser.add_argument("--dst_cage", type=str, default=None)
    parser.add_argument("--override_cam", type=Path, default=None)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--store_proxy", action="store_true")

    args = get_combined_args(parser)
    model.gs_type = args.gs_type
    model.num_splats = args.num_splats
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(
        model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test,
        getattr(args, "src_cage", None), getattr(args, "dst_cage", None), getattr(args, "override_cam", None), args.benchmark, args.store_proxy
    )