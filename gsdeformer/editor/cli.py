import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torchvision.utils
import trimesh
from omegaconf import MISSING, OmegaConf
import open3d
import open3d.camera

# noinspection PyUnresolvedReferences
import gsdeformer.hack_add_gs3d_to_sys_path
from gs3d.arguments import PipelineParams
from gs3d.gaussian_renderer import render
from gsdeformer.deform.algorithm_ui import build_deformer
from gsdeformer.editor.__main__ import WHITE, BLACK
from gsdeformer.editor.utils_camera import cvt_camera_info_to_camera, cvt_camera_o3d_to_3dgs
from gsdeformer.editor.utils_data import load_model, list_cameras
from gsdeformer.utils_config import parse_args
from loguru import logger as log

@dataclass
class EditorCLIArguments:
    model_path: Path = MISSING  # path to trained GS3D folder
    iteration: int = -1  # the iter of model to load, -1 means to pick the latest

    algorithm: str = "all" # ablation switch, use full algorithm or "mean_only" algorithm
    cage_path: Path = MISSING # path to cage location
    dst_cage_path: Path = MISSING # path to deformed cage location

    cam_idx: int = 0 # render camera index
    cam_ckpt: Optional[Path] = None
    source_path: Path = MISSING # path to dataset folder
    white_bg: bool = True # white background in rendering?

    output_path: Path = MISSING # path to place output

def unpicklize_camera_param(camera_param):
    intrin, extrin = camera_param
    width, height, intrinsic_matrix = intrin
    intrin = open3d.camera.PinholeCameraIntrinsic(width, height, intrinsic_matrix)
    return (intrin, extrin)

@torch.no_grad()
def main(args=None):
    args = parse_args(EditorCLIArguments, args)
    log.debug("running wih configuration: \n{}", OmegaConf.to_container(args))

    log.debug("loading model, cage & cameras")
    model = load_model(args.model_path, iteration=args.iteration)
    src_cage = trimesh.load_mesh(args.cage_path)
    dst_cage = trimesh.load(args.dst_cage_path)
    cameras = list_cameras(args.source_path)
    cameras = sorted(cameras, key=lambda c: c.image_name)
    if args.cam_ckpt:
        cam = unpicklize_camera_param(torch.load(args.cam_ckpt))
        camera = cvt_camera_o3d_to_3dgs(cam)
    else:
        camera = cvt_camera_info_to_camera(cameras[args.cam_idx])

    log.debug("running deformation")
    deformer = build_deformer(args.algorithm, model, src_cage, pbar=True)
    deformed = deformer(dst_cage)

    log.debug("running rendering")
    pipe = PipelineParams(argparse.ArgumentParser())  # mock pipeline params
    bg_color = torch.tensor(WHITE if args.white_bg else BLACK, dtype=torch.float32, device="cuda")
    image = render(camera, model, pipe, bg_color)["render"]
    image = torch.clamp(image, 0.0, 1.0)
    cov3d = getattr(deformed, "precomp_cov", None)
    image_deformed = render(camera, deformed, pipe, bg_color, cov3d=cov3d)["render"]
    image_deformed = torch.clamp(image_deformed, 0.0, 1.0)

    log.debug("saving")
    args.output_path.mkdir(parents=True, exist_ok=True)
    filename = f"{args.model_path.name}_{args.cam_idx}_{args.algorithm}"
    torchvision.utils.save_image(image, args.output_path / f"{filename}_og.png")
    torchvision.utils.save_image(image_deformed, args.output_path / f"{filename}_deformed.png")


if __name__ == '__main__':
    main()

