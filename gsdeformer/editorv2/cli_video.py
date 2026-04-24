from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch
import tqdm
import PIL.Image as Image
from loguru import logger as log
from omegaconf import MISSING, OmegaConf

# noinspection PyUnresolvedReferences
import gsdeformer.hack_add_gs3d_to_sys_path
from gsdeformer.deform.algorithm_ui import build_deformer
from gsdeformer.editorv2.utils_data import load_model
from gsdeformer.editorv2.cli import build_renderer, RenderOptions
from gsdeformer.editorv2.cli_video_animations import build_cameras, build_cages
from gsdeformer.utils_config import parse_args


@dataclass
class VideoArguments:
    model_path: Path = MISSING  # path to trained GS3D folder
    iteration: int = -1  # the iter of model to load, -1 means to pick the latest

    cameras: Dict[str, Any] = MISSING # root for camera config, see build_cameras() for details
    cages: Dict[str, Any] = MISSING # root for cage config, see build_cages() for details

    algorithm: str = "all" # ablation switch, use full algorithm or "mean_only" algorithm

    white_bg: bool = True # white background in rendering?
    output_path: Path = MISSING # path to place output


@torch.no_grad()
def main(args=None):
    import taichi as ti
    ti.init(ti.gpu)

    args = parse_args(VideoArguments, args)
    log.debug("running wih configuration: \n{}", OmegaConf.to_container(args))

    log.debug("loading cameras, model & cage")
    model = load_model(args.model_path, iteration=args.iteration)
    cameras = build_cameras(args.cameras)
    length = len(cameras)
    cages = build_cages(args.cages)
    src_cage = next(cages)

    log.debug("preparing deformation")
    deformer = build_deformer(args.algorithm, model, src_cage, pbar=False)

    log.debug("running rendering")
    it = enumerate(zip(cameras, cages))
    it = tqdm.tqdm(it, total=length, desc="rendering")
    args.output_path.mkdir(parents=True, exist_ok=True)
    renderer = None
    for idx, (cam, cage) in it:

        if renderer is None:
            renderer = build_renderer(cam, model, src_cage, cage, args.white_bg)

        deformed = deformer(cage)

        image_deformed = renderer(
            cam, deformed, src_cage, cage,
            opt=RenderOptions(gaussian=True, src_cage=True, dst_cage=True)
        )

        Image.fromarray(image_deformed).save(args.output_path / f"{idx:06}.png")


if __name__ == '__main__':
    main()

