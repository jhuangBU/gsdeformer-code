import argparse
import itertools
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Tuple, List

import numpy as np
import torch
import torchvision.utils
import tqdm
import trimesh
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from loguru import logger as log
from omegaconf import MISSING, OmegaConf

# noinspection PyUnresolvedReferences
import gsdeformer.hack_add_gs3d_to_sys_path
from gs3d.arguments import PipelineParams
from gs3d.gaussian_renderer import render
from gs3d.scene.cameras import Camera
from gs3d.scene.colmap_loader import read_intrinsics_text, read_extrinsics_text, qvec2rotmat
from gs3d.scene.dataset_readers import readColmapCameras, CameraInfo
from gs3d.utils.graphics_utils import focal2fov
from gsdeformer.deform.algorithm_ui import build_deformer
from gsdeformer.editor.__main__ import WHITE, BLACK
from gsdeformer.editor.utils_camera import cvt_camera_info_to_camera
from gsdeformer.editor.utils_data import load_model, list_cameras
from gsdeformer.utils_config import parse_args


@dataclass
class VideoArguments:
    source_path: Path = MISSING # path to dataset folder

    model_path: Path = MISSING  # path to trained GS3D folder
    iteration: int = -1  # the iter of model to load, -1 means to pick the latest

    algorithm: str = "all" # ablation switch, use full algorithm or "mean_only" algorithm
    cage_path: Path = MISSING # path to cage location
    dst_cage_path: Path = MISSING # path to deformed cage location
    deform_cycle: int = 40 # how many frames will the deform cycle
    white_bg: bool = True # white background in rendering?

    animation_variant: str = MISSING # variant used for animation

    output_path: Path = MISSING # path to place output

# bounded iterator, yields cameras
CameraAnimation = Iterator[Camera]

# maybe unbounded iterator, yields cages, with the first one being src
CageAnimation = Iterator[trimesh.Trimesh]

def deforming_nerf_cameras(args: VideoArguments) -> Tuple[int, CameraAnimation]:
    cameras = list_cameras(args.source_path)
    cameras = sorted(cameras, key=lambda c: c.image_name)
    cameras = cameras[100:]
    cameras = [cvt_camera_info_to_camera(cam) for cam in cameras]
    return len(cameras), iter(cameras)

def deforming_nerf_cages(args: VideoArguments) -> CageAnimation:
    src_cage: trimesh.Trimesh = trimesh.load_mesh(args.cage_path)
    dst_cages = []
    dst_cage_path = args.dst_cage_path
    dst_cages.append(trimesh.load(dst_cage_path))
    i = 0
    while True:
        p = dst_cage_path.parent / (dst_cage_path.stem + f"{i}" + dst_cage_path.suffix)
        print("searching", p)
        if not p.exists():
            break
        else:
            dst_cages.append(trimesh.load(p))
            i += 1

    yield src_cage
    while True:
        for dst_cage in dst_cages:
            for weight in emit_interpolate_weights(args.deform_cycle, cycle=False):
                d = dst_cage.copy()
                d.vertices = d.vertices * weight + src_cage.vertices * (1 - weight)
                yield d

def emit_interpolate_weights(cycles: int, cycle: bool = True) -> Iterator[int]:
    assert cycles % 2 == 0
    s = np.linspace(0.0, 1.0, cycles // 2)
    seq = np.concatenate([s, np.flip(s, 0)], axis=0)
    if cycle:
        return itertools.cycle(seq)
    else:
        return iter(seq)

def mipnerf_cameras(args: VideoArguments) -> Tuple[int, CameraAnimation]:
    cameras_extrinsic_file = os.path.join(str(args.source_path), "images.txt")
    cameras_intrinsic_file = os.path.join(str(args.source_path), "cameras.txt")
    cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    cam_infos = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics)

    cams = []
    frame_count = 20
    for c in range(len(cam_infos)-1):
        s = cam_infos[c]
        d = cam_infos[c+1]

        cis = interpolate_cameras(frame_count, s, d)

        for c in cis:
            cams.append(cvt_camera_info_to_camera(c))

    return len(cams), iter(cams)

def interpolate_cameras(frame_count: int, s: CameraInfo, d: CameraInfo) -> List[CameraInfo]:
    weights = np.linspace(0.0, 1.0, frame_count, endpoint=False)
    slerp = Slerp(times=[0.0, 1.0], rotations=R.from_matrix(np.stack([s.R, d.R])))
    rs = slerp(weights)
    rs = rs.as_matrix()
    ts = [s.T * (1 - i) + d.T * i for i in weights]
    cis = []
    for r, t in zip(rs, ts):
        ci = CameraInfo(
            uid=s.uid,
            R=r,
            T=t,
            FovY=s.FovY,
            FovX=s.FovX,
            image=None,
            image_path=None,
            image_name=s.image_name,
            width=s.width,
            height=s.height
        )
        cis.append(ci)
    return cis

def readColmapCameras(cam_extrinsics, cam_intrinsics):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.basename(extr.name)
        image_name = os.path.basename(image_path).split(".")[0]

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=None,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)

    return cam_infos

def mipnerf_cages(args: VideoArguments) -> CageAnimation:
    return deforming_nerf_cages(args)

def dtu_cameras(args: VideoArguments) -> Tuple[int, CameraAnimation]:
    cameras_extrinsic_file = os.path.join(str(args.source_path), "images.txt")
    cameras_intrinsic_file = os.path.join(str(args.source_path), "cameras.txt")
    cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    cam_infos = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics)

    cams = []
    frame_count = 30
    for c in range(len(cam_infos)-1):
        s = cam_infos[c]
        d = cam_infos[c+1]

        cis = interpolate_cameras(frame_count, s, d)

        for c in cis:
            cams.append(cvt_camera_info_to_camera(c))
    cameras = cams

    return len(cameras), iter(cameras)

def dtu_cages(args: VideoArguments) -> CageAnimation:
    return deforming_nerf_cages(args)

VARIANTS = {
    "deforming-nerf": (deforming_nerf_cameras, deforming_nerf_cages),
    "dtu": (dtu_cameras, dtu_cages),
    "mipnerf360": (mipnerf_cameras, mipnerf_cages)
}

@torch.no_grad()
def main(args=None):
    args = parse_args(VideoArguments, args)
    log.debug("running wih configuration: \n{}", OmegaConf.to_container(args))

    log.debug("loading cameras, model & cage")
    model = load_model(args.model_path, iteration=args.iteration)
    cameras_factory, cages_factory = VARIANTS[args.animation_variant]
    length, cameras = cameras_factory(args)
    cages = cages_factory(args)
    src_cage = next(cages)

    log.debug("preparing deformation")
    deformer = build_deformer(args.algorithm, model, src_cage, pbar=True)

    log.debug("running rendering")
    pipe = PipelineParams(argparse.ArgumentParser())  # mock pipeline params
    bg_color = torch.tensor(WHITE if args.white_bg else BLACK, dtype=torch.float32, device="cuda")
    it = enumerate(zip(cameras, cages))
    it = tqdm.tqdm(it, total=length, desc="rendering")
    args.output_path.mkdir(parents=True, exist_ok=True)
    for idx, (cam, cage) in it:
        deformed = deformer(cage)

        image_deformed = render(cam, deformed, pipe, bg_color)["render"]
        image_deformed = torch.clamp(image_deformed, 0.0, 1.0)

        torchvision.utils.save_image(image_deformed, args.output_path / f"{idx:06}.png")


if __name__ == '__main__':
    main()

