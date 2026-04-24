from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List
from unittest.mock import Mock

import PIL.Image as Image
import dill
import numpy as np
import open3d as o3d
import torch
import trimesh
from loguru import logger as log
from omegaconf import MISSING, OmegaConf

# noinspection PyUnresolvedReferences
import gsdeformer.hack_add_gs3d_to_sys_path
from scene.dataset_readers import CameraInfo
from scene.gaussian_model import GaussianModel
from gsdeformer.deform.algorithm_ui import build_deformer
from gsdeformer.editor.utils_camera import cvt_camera_info_to_camera, cvt_camera_3dgs_to_o3d_intrin
from gsdeformer.editor.utils_data import load_model, list_cameras
from gsdeformer.editorv2.__main__ import CompositeRenderer, GaussianRenderer, CageRenderer
from gsdeformer.editorv2.utils_camera import cvt_camera_3dgs_to_o3d_extrin
from gsdeformer.editorv2.utils_color import BLUE, RED
from gsdeformer.utils_config import parse_args
from gsdeformer.utils_ui import ValueEmitter


@dataclass
class EditorCLIArguments:
    model_path: Path = MISSING  # path to trained GS3D folder
    iteration: int = -1  # the iter of model to load, -1 means to pick the latest

    algorithm: str = "all" # ablation switch, use full algorithm or "mean_only" algorithm
    splitting: bool = True # ablation switch, enable splitting?
    cage_path: Path = MISSING # path to cage location
    dst_cage_path: Path = MISSING # path to deformed cage location

    cam_idx: int = 0 # render camera index, use -1 to render all cameras
    cam_ckpt: Optional[Path] = None # overriding camera parameter, if specified, will use this instead
    source_path: Path = MISSING # path to dataset folder
    white_bg: bool = True # white background in rendering?
    use_camera_resolution: bool = False # use camera's native resolution instead of 512x512

    save_model: bool = False # save the model after editing?
    expname: str = "" # name of the experiment for output image
    output_path: Path = MISSING # path to place output

@dataclass
class RenderOptions:
    gaussian: bool = False
    dst_cage: bool = False
    src_cage: bool = False
    vcloud: bool = False

def build_renderer(cam: CameraInfo, model: GaussianModel, src: trimesh.Trimesh, dst: trimesh.Trimesh, white_bg:bool, height=512, width=512):

    def _cage_to_display(cage, color):
        s = o3d.geometry.LineSet.create_from_triangle_mesh(cage.as_open3d)
        s = s.paint_uniform_color(color[:, np.newaxis])
        return s

    def _cage_to_vcloud(cage):
        vcloud = o3d.geometry.PointCloud()
        vcloud.points = o3d.utility.Vector3dVector(np.asarray(cage.vertices))
        vcloud = vcloud.paint_uniform_color(BLUE[:, np.newaxis])
        return vcloud

    model = ValueEmitter(model)
    vcloud = ValueEmitter(_cage_to_vcloud(dst))
    src, dst = ValueEmitter(_cage_to_display(src, BLUE)), ValueEmitter(_cage_to_display(dst, RED))

    gr = GaussianRenderer(model, white_bg=white_bg)
    cr = CageRenderer([src, dst], vcloud)
    r = CompositeRenderer(gr, cr)

    window = Mock()
    window.scaling = 1.0

    # mock render
    intrin = cvt_camera_3dgs_to_o3d_intrin(cam, height=height, width=width)
    extrin = cvt_camera_3dgs_to_o3d_extrin(cam)
    r.render(window, intrin, extrin)

    def _render(cam: CameraInfo, m: GaussianModel, s: trimesh.Trimesh, d: trimesh.Trimesh, opt: RenderOptions=None):
        if not opt:
            opt = RenderOptions()

        gr.set_visible(opt.gaussian)
        cr.set_cage_visible(0, opt.src_cage)
        cr.set_cage_visible(1, opt.dst_cage)
        cr.set_vertex_cloud_visible(opt.vcloud)

        model.emit(m)
        src.emit(_cage_to_display(s, BLUE))
        dst.emit(_cage_to_display(d, RED))

        intrin = cvt_camera_3dgs_to_o3d_intrin(cam, height=height, width=width)
        extrin = cvt_camera_3dgs_to_o3d_extrin(cam)

        rgb, *rest = r.render(window, intrin, extrin)

        return rgb

    return _render

def build_composite_renderer(cam: CameraInfo, model: GaussianModel, src_dst_cages: List[Tuple[trimesh.Trimesh, trimesh.Trimesh]], white_bg:bool):
    """
    TODO: merge it with build_renderer()
    """

    def _cage_to_display(cage, color):
        s = o3d.geometry.LineSet.create_from_triangle_mesh(cage.as_open3d)
        s = s.paint_uniform_color(color[:, np.newaxis])
        return s

    model = ValueEmitter(model)
    src_cages = []
    dst_cages = []
    for src, dst in src_dst_cages:
        src_cages.append(ValueEmitter(_cage_to_display(src, BLUE)))
        dst_cages.append(ValueEmitter(_cage_to_display(dst, RED)))

    gr = GaussianRenderer(model, white_bg=white_bg)
    cr = CageRenderer([*src_cages, *dst_cages], None)
    r = CompositeRenderer(gr, cr)

    window = Mock()
    window.scaling = 1.0

    # mock render
    intrin = cvt_camera_3dgs_to_o3d_intrin(cam)
    extrin = cvt_camera_3dgs_to_o3d_extrin(cam)
    r.render(window, intrin, extrin)

    def _render(cam: CameraInfo, m: GaussianModel, sds: List[Tuple[trimesh.Trimesh, trimesh.Trimesh]], opt: RenderOptions=None):
        gr.set_visible(opt.gaussian)
        for idx, _ in enumerate(src_cages):
            cr.set_cage_visible(idx, opt.src_cage)
        for idx, _ in enumerate(dst_cages):
            cr.set_cage_visible(len(src_cages)+idx, opt.dst_cage)
        assert opt.vcloud == False, "vertice cloud display is not supported now!"

        model.emit(m)

        for src, dst, (s,d) in zip(src_cages, dst_cages, sds):
            src.emit(_cage_to_display(s, BLUE))
            dst.emit(_cage_to_display(d, RED))

        intrin = cvt_camera_3dgs_to_o3d_intrin(cam, height=cam.height, width=cam.width)
        extrin = cvt_camera_3dgs_to_o3d_extrin(cam)

        rgb, *rest = r.render(window, intrin, extrin)

        return rgb

    return _render

@torch.no_grad()
def main(args=None):
    import taichi as ti
    ti.init(ti.gpu)

    args = parse_args(EditorCLIArguments, args)
    log.debug("running wih configuration: \n{}", OmegaConf.to_container(args))

    log.debug("loading model, cage & cameras")
    model = load_model(args.model_path, iteration=args.iteration)
    src_cage = trimesh.load_mesh(args.cage_path)
    dst_cage = trimesh.load(args.dst_cage_path)

    # Load cameras
    all_cameras = list_cameras(args.source_path)
    all_cameras = sorted(all_cameras, key=lambda c: c.image_name)

    # Determine which cameras to render
    if args.cam_idx == -1:
        cameras_to_render = all_cameras
    else:
        if args.cam_ckpt:
            camera = torch.load(args.cam_ckpt, pickle_module=dill)
        else:
            camera = all_cameras[args.cam_idx]
        cameras_to_render = [camera]

    # Determine rendering resolution
    render_height = cameras_to_render[0].height if args.use_camera_resolution else 512
    render_width = cameras_to_render[0].width if args.use_camera_resolution else 512
    renderer = build_renderer(cameras_to_render[0], model, src_cage, dst_cage, args.white_bg, render_height, render_width)

    log.debug("running deformation")
    kwargs = {}
    if args.save_model:
        # uncomment for render == saving
        # kwargs["return_precomp_cov"] = False
        kwargs["compute_rot_scaling"] = True
    if not args.splitting:
        kwargs["splitting"] = False
    deformer = build_deformer(
        args.algorithm, model, src_cage, pbar=True,
        **kwargs
    )
    deformed = deformer(dst_cage)

    log.debug("running rendering")
    args.output_path.mkdir(parents=True, exist_ok=True)
    split_suffix = "_no_split" if not args.splitting else ""

    # Single rendering loop for both single and multiple cameras
    for camera in cameras_to_render:
        image = renderer(camera, model, src_cage, dst_cage, opt=RenderOptions(gaussian=True))
        image_deformed = renderer(camera, deformed, src_cage, dst_cage, opt=RenderOptions(gaussian=True))
        image_src = renderer(camera, deformed, src_cage, dst_cage, opt=RenderOptions(src_cage=True))
        image_dst = renderer(camera, deformed, src_cage, dst_cage, opt=RenderOptions(dst_cage=True))

        log.debug("saving")
        # Filename: use camera.image_name directly for test set
        if args.cam_idx == -1:
            filename = camera.image_name
        else:
            filename = f"{args.expname if args.expname else args.model_path.name}_{args.cam_idx}_{args.algorithm}{split_suffix}"

        Image.fromarray(image).save(args.output_path / f"{filename}_og.png")
        Image.fromarray(image_deformed).save(args.output_path / f"{filename}_deformed.png")
        Image.fromarray(image_src).save(args.output_path / f"{filename}_src.png")
        Image.fromarray(image_dst).save(args.output_path / f"{filename}_dst.png")
        if args.save_model:
            deformed.save_ply(str(args.output_path / f"{filename}_deformed.ply"))


if __name__ == '__main__':
    main()

