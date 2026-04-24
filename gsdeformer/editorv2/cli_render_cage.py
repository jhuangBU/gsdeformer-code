"""
helper script for rendering cage from GSDeformer
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
from unittest.mock import Mock

import PIL.Image as Image
import dill
import numpy as np
import torch
from loguru import logger as log
from omegaconf import MISSING, OmegaConf
# noinspection PyUnresolvedReferences
import gsdeformer.hack_add_gs3d_to_sys_path
from scene.dataset_readers import CameraInfo
import open3d.geometry as ogeo
import open3d.io as oio

from gsdeformer.editor.utils_camera import cvt_camera_3dgs_to_o3d_intrin
from gsdeformer.editorv2.__main__ import CageRenderer
from gsdeformer.editorv2.utils_camera import cvt_camera_3dgs_to_o3d_extrin
from gsdeformer.utils_config import parse_args
from gsdeformer.utils_ui import ValueEmitter


@dataclass
class EditorCLIArguments:
    cage_path: Path = MISSING # path to cage
    cam_ckpt: Optional[Path] = None # overriding camera parameter, if specified, will use this instead
    output_path: Path = MISSING # path to output image


def build_renderer(cam: CameraInfo, tet: ogeo.LineSet):
    tetem = ValueEmitter(tet)
    tet.paint_uniform_color([0, 0, 1]) # blue
    vcloud = ogeo.PointCloud(points=tet.points)
    vcloud.paint_uniform_color([1, 0, 0]) # red
    vcloudem = ValueEmitter(vcloud)

    r = CageRenderer(cages=[tetem], vcloud=vcloudem)

    window = Mock()
    window.scaling = 1.0

    # mock render
    intrin = cvt_camera_3dgs_to_o3d_intrin(cam, height=512, width=512)
    extrin = cvt_camera_3dgs_to_o3d_extrin(cam)
    r.render(window, intrin, extrin)

    def _render(cam: CameraInfo, t: ogeo.LineSet):
        tetem.emit(t)

        intrin = cvt_camera_3dgs_to_o3d_intrin(cam, height=512, width=512)
        extrin = cvt_camera_3dgs_to_o3d_extrin(cam)

        rgb, *rest = r.render(window, intrin, extrin)

        return rgb

    return _render


@torch.no_grad()
def main(args=None):
    args = parse_args(EditorCLIArguments, args)
    log.debug("running with configuration: \n{}", OmegaConf.to_container(args))

    log.debug("loading tet & camera")
    cage = oio.read_triangle_mesh(str(args.cage_path))
    tet = ogeo.LineSet.create_from_triangle_mesh(cage)
    camera = torch.load(args.cam_ckpt, pickle_module=dill)
    renderer = build_renderer(camera, tet)

    log.debug("running rendering")
    image = renderer(camera, tet)

    log.debug("saving")
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(str(args.output_path))


if __name__ == '__main__':
    main()

