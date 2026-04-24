from dataclasses import dataclass, field
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
from gsdeformer.deform.model_util import merge_model
from gsdeformer.editorv2.utils_data import load_model
from gsdeformer.editorv2.cli import RenderOptions, build_composite_renderer
from gsdeformer.editorv2.cli_video_animations import build_cameras, build_cages
from gsdeformer.utils_config import parse_args


@dataclass
class ModelEntry:
    model_path: Path = MISSING  # path to trained GS3D folder
    iteration: int = -1  # the iter of model to load, -1 means to pick the latest
    cages: Dict[str, Any] = field(default_factory=dict) # root for cage config, see build_cages() for details

@dataclass
class VideoArguments:
    cameras: Dict[str, Any] = MISSING # root for camera config, see build_cameras() for details
    models: Dict[str, ModelEntry] = MISSING # animated models

    algorithm: str = "all" # ablation switch, use full algorithm or "mean_only" algorithm

    white_bg: bool = True # white background in rendering?
    output_path: Path = MISSING # path to place output


@torch.no_grad()
def main(args=None):
    import taichi as ti
    ti.init(ti.gpu)

    args = parse_args(VideoArguments, args)
    log.debug("running wih configuration: \n{}", OmegaConf.to_container(args))

    log.debug("loading cameras")
    cameras = build_cameras(args.cameras)
    length = len(cameras)

    log.debug("preparing deformation")
    animated_model = []
    for k,v in args.models.items():
        model = load_model(v.model_path, iteration=v.iteration)
        cages = build_cages(v.cages)
        src_cage = next(cages)
        deformer = build_deformer(args.algorithm, model, src_cage, pbar=False)
        animated_model.append({"key": k, "model": model, "src_cage": src_cage, "cages": cages, "deformer": deformer})

    log.debug("running rendering")
    it = enumerate(cameras)
    it = tqdm.tqdm(it, total=length, desc="rendering")
    args.output_path.mkdir(parents=True, exist_ok=True)
    renderer = None
    for idx, cam in it:

        if renderer is None:
            cages = [(m["src_cage"], m["src_cage"]) for m in animated_model]
            model = None
            for m in animated_model:
                model = m["model"] if model is None else merge_model(model, m["model"])
            renderer = build_composite_renderer(cam, model, cages, args.white_bg)

        for m in animated_model:
            cages = next(m["cages"])
            m["model"] = m["deformer"](cages)

        cages = [(m["src_cage"], m["src_cage"]) for m in animated_model]
        model = None
        for m in animated_model:
            model = m["model"] if model is None else merge_model(model, m["model"])
        image_deformed = renderer(
            cam, model, cages,
            opt=RenderOptions(gaussian=True)
        )

        Image.fromarray(image_deformed).save(args.output_path / f"{idx:06}.png")


if __name__ == '__main__':
    main()

