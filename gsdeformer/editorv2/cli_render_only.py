from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from unittest.mock import Mock

import PIL.Image as Image
import dill
import torch
from loguru import logger as log
from omegaconf import MISSING, OmegaConf
# noinspection PyUnresolvedReferences
import gsdeformer.hack_add_gs3d_to_sys_path
from scene.dataset_readers import CameraInfo
from scene.gaussian_model import GaussianModel

from gsdeformer.editor.utils_camera import cvt_camera_3dgs_to_o3d_intrin
from gsdeformer.editor.utils_data import load_model, list_cameras
from gsdeformer.editorv2.__main__ import GaussianRenderer
from gsdeformer.editorv2.utils_camera import cvt_camera_3dgs_to_o3d_extrin
from gsdeformer.utils_config import parse_args
from gsdeformer.utils_ui import ValueEmitter


@dataclass
class EditorCLIArguments:
    model_path: Path = MISSING  # path to trained GS3D folder
    iteration: int = -1  # the iter of model to load, -1 means to pick the latest

    cam_idx: int = 0 # render camera index
    cam_ckpt: Optional[Path] = None # overriding camera parameter, if specified, will use this instead
    source_path: Path = MISSING # path to dataset folder
    white_bg: bool = True # white background in rendering?

    expname: str = "" # name of the experiment for output image
    output_path: Path = MISSING # path to place output

@dataclass
class RenderOptions:
    gaussian: bool = False
    dst_cage: bool = False
    src_cage: bool = False
    vcloud: bool = False

def build_renderer(cam: CameraInfo, model: GaussianModel, white_bg:bool):

    model = ValueEmitter(model)

    r = GaussianRenderer(model, white_bg=white_bg)

    window = Mock()
    window.scaling = 1.0

    # mock render
    intrin = cvt_camera_3dgs_to_o3d_intrin(cam, height=512, width=512)
    extrin = cvt_camera_3dgs_to_o3d_extrin(cam)
    r.render(window, intrin, extrin)

    def _render(cam: CameraInfo, m: GaussianModel, opt: RenderOptions=None):
        if not opt:
            opt = RenderOptions()

        r.set_visible(opt.gaussian)

        model.emit(m)

        intrin = cvt_camera_3dgs_to_o3d_intrin(cam, height=512, width=512)
        extrin = cvt_camera_3dgs_to_o3d_extrin(cam)

        rgb, *rest = r.render(window, intrin, extrin)

        return rgb

    return _render

@torch.no_grad()
def main(args=None):
    args = parse_args(EditorCLIArguments, args)
    log.debug("running with configuration: \n{}", OmegaConf.to_container(args))

    log.debug("loading model & cameras")
    model = load_model(args.model_path, iteration=args.iteration)
    cameras = list_cameras(args.source_path)
    cameras = sorted(cameras, key=lambda c: c.image_name)
    if args.cam_ckpt:
        camera = torch.load(args.cam_ckpt, pickle_module=dill)
    else:
        camera = cameras[args.cam_idx]
    renderer = build_renderer(camera, model, args.white_bg)

    log.debug("running rendering")
    image = renderer(camera, model, opt=RenderOptions(gaussian=True))

    log.debug("saving")
    args.output_path.mkdir(parents=True, exist_ok=True)
    filename = f"{args.expname if args.expname else args.model_path.name}_{args.cam_idx}"
    Image.fromarray(image).save(args.output_path / f"{filename}.png")

if __name__ == '__main__':
    main()

