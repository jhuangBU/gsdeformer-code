import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torchvision.utils
import trimesh
from omegaconf import MISSING, OmegaConf

# noinspection PyUnresolvedReferences
import gsdeformer.hack_add_gs3d_to_sys_path
from gs3d.arguments import PipelineParams
from gs3d.gaussian_renderer import render
from gsdeformer.deform.algorithm_ui import build_deformer
from gsdeformer.editor.__main__ import WHITE, BLACK
from gsdeformer.editor.utils_camera import cvt_camera_info_to_camera
from gsdeformer.editor.utils_data import load_model, list_cameras
from gsdeformer.utils_config import parse_args
from loguru import logger as log


@dataclass
class EditorCLIBenchmarkArguments:
    model_path: Path = MISSING  # path to trained GS3D folder
    iteration: int = -1  # the iter of model to load, -1 means to pick the latest

    algorithm: str = "all" # ablation switch, use full algorithm or "mean_only" algorithm
    cage_path: Path = MISSING # path to cage location
    dst_cage_path: Path = MISSING # path to deformed cage location

    source_path: Path = None # path to dataset folder

    warmup_iter: int = 2
    measure_iter: int = 3

    output_path: Path = MISSING  # path to place output


@torch.no_grad()
def main(args=None):
    import taichi as ti
    ti.init(ti.gpu)

    args = parse_args(EditorCLIBenchmarkArguments, args)
    log.debug("running wih configuration: \n{}", OmegaConf.to_container(args))

    log.debug("loading model, cage & cameras")
    model = load_model(args.model_path, iteration=args.iteration)
    src_cage = trimesh.load_mesh(args.cage_path)
    dst_cage = trimesh.load(args.dst_cage_path)
    if args.source_path:
        cameras = list_cameras(args.source_path)
        cameras = sorted(cameras, key=lambda c: c.image_name)
        camera = cameras[0]
    else:
        log.warning("source_path not provided, skipping rendering")
        camera = None

    log.debug(f"running deformation on {model.get_xyz.shape[0]} gaussians")
    log.debug("warming up")
    torch.cuda.synchronize()
    for _ in range(args.warmup_iter):
        deformer = build_deformer(args.algorithm, model, src_cage)
        deformed = deformer(dst_cage)
    torch.cuda.synchronize()

    log.debug("measuring")
    flag_begin = torch.cuda.Event(enable_timing=True)
    flag_preprocessed = torch.cuda.Event(enable_timing=True)
    flag_finished = torch.cuda.Event(enable_timing=True)
    render_finished = torch.cuda.Event(enable_timing=True)
    preprocess_times = []
    finish_times = []
    render_times = []
    for _ in range(args.measure_iter):
        flag_begin.record()
        deformer = build_deformer(args.algorithm, model, src_cage)
        flag_preprocessed.record()
        deformed = deformer(dst_cage)
        flag_finished.record()
        if camera:
            pipe = PipelineParams(argparse.ArgumentParser())  # mock pipeline params
            bg_color = torch.tensor(WHITE, dtype=torch.float32, device="cuda")
            cov3d = getattr(deformed, "precomp_cov", None)
            render(camera, deformed, pipe, bg_color, cov3d=cov3d)["render"]
        render_finished.record()
        torch.cuda.synchronize()
        preprocess_times.append(flag_begin.elapsed_time(flag_preprocessed))
        finish_times.append(flag_preprocessed.elapsed_time(flag_finished))
        render_times.append(flag_finished.elapsed_time(render_finished))

    results = {
        "gaussian_count": model.get_xyz.shape[0],
        "preprocess_times": preprocess_times,
        "finish_times": finish_times,
        "render_times": render_times,
        "preprocess_time_avg": sum(preprocess_times) / len(preprocess_times),
        "finish_time_avg": sum(finish_times) / len(finish_times),
        "render_time_avg": sum(render_times) / len(render_times)
    }

    log.debug("writing results")
    filename = f"{args.model_path.name}_{args.algorithm}"
    (args.output_path / f"{filename}_benchmark.json").write_text(json.dumps(results, indent=4), encoding="utf-8")


if __name__ == '__main__':
    main()

