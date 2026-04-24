"""
helper script for rendering tetrahedral mesh from VR-GS
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
import open3d.utility as outil

from gsdeformer.editor.utils_camera import cvt_camera_3dgs_to_o3d_intrin
from gsdeformer.editorv2.__main__ import CageRenderer
from gsdeformer.editorv2.utils_camera import cvt_camera_3dgs_to_o3d_extrin
from gsdeformer.utils_config import parse_args
from gsdeformer.utils_ui import ValueEmitter


@dataclass
class EditorCLIArguments:
    tet_path: Path = MISSING # path to tetrahedral mesh
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


def read_tet_txt(file: Path) -> Tuple[np.ndarray, np.ndarray]:
    with open(file, 'r') as file:
        # Read header line with node and element counts
        first_line = file.readline().strip().split()
        num_nodes = int(first_line[0])
        num_elems = int(first_line[1])

        # Initialize arrays
        nodes = np.zeros((num_nodes, 3))  # Assuming 3D coordinates
        elems = np.zeros((num_elems, 4), dtype=int)  # Assuming tetrahedral elements

        # Read node coordinates
        for i in range(num_nodes):
            line = file.readline().strip().split()
            nodes[i, :] = [float(val) for val in line]

        # Read element connectivity
        for i in range(num_elems):
            line = file.readline().strip().split()
            elems[i, :] = [int(val) for val in line]

    return nodes, elems


def tet_to_line_set(tet: Tuple[np.ndarray, np.ndarray]) -> ogeo.LineSet:
    nodes, elems = tet

    # Create LineSet
    lines = []
    for tet in elems:
        # Add lines for each edge of tetrahedron
        lines.extend([
            [tet[0], tet[1]], [tet[1], tet[2]], [tet[2], tet[0]],
            [tet[0], tet[3]], [tet[1], tet[3]], [tet[2], tet[3]]
        ])

    line_set = ogeo.LineSet()
    line_set.points = outil.Vector3dVector(nodes)
    line_set.lines = outil.Vector2iVector(lines)

    return line_set


@torch.no_grad()
def main(args=None):
    args = parse_args(EditorCLIArguments, args)
    log.debug("running with configuration: \n{}", OmegaConf.to_container(args))

    log.debug("loading tet & camera")
    tet = read_tet_txt(args.tet_path)
    tet = tet_to_line_set(tet)
    camera = torch.load(args.cam_ckpt, pickle_module=dill)
    renderer = build_renderer(camera, tet)

    log.debug("running rendering")
    image = renderer(camera, tet)

    log.debug("saving")
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(str(args.output_path))


if __name__ == '__main__':
    main()

