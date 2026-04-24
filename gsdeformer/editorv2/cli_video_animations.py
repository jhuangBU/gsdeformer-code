import itertools
import os
from dataclasses import dataclass, MISSING
from pathlib import Path
from typing import Iterator, List, ClassVar, Dict, Any
from loguru import logger as log
import numpy as np
import torch
import trimesh
from omegaconf import OmegaConf
from scipy.spatial.transform import Slerp, Rotation as R

from gs3d.scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat
from gs3d.scene.dataset_readers import CameraInfo, readColmapCameras
from gs3d.utils.graphics_utils import focal2fov
from gsdeformer.editor.utils_data import list_cameras

# bounded iterator, yields cameras
CameraAnimation = List[CameraInfo]

# maybe unbounded iterator, yields cages, with the first one being src
CageAnimation = Iterator[trimesh.Trimesh]

@dataclass
class GSDatasetCameras:
    """
    use cameras loaded from the dataset folder
    """
    TYPE: ClassVar[str] = "gs_dataset"
    source_path: Path = MISSING
    camera_start: int = 100
    interpolate: int = 0


def gs_dataset_cameras(args: GSDatasetCameras) -> CameraAnimation:
    cameras = list_cameras(args.source_path)
    cameras = sorted(cameras, key=lambda c: c.image_name)
    cameras = cameras[args.camera_start:]
    if args.interpolate:
        interpolated_cameras = []
        for i in range(len(cameras) - 1):
            first = cameras[i]
            second = cameras[i + 1]
            interpolated = interpolate_cameras(args.interpolate, first, second)
            interpolated_cameras.extend(interpolated)
        cameras = interpolated_cameras
    return cameras

@dataclass
class CageFolderCages:
    """
    read cages in folder, assuming cage_path is a.ply, dst_cage_path is b.ply, it loads a.ply, b0.ply, b1.ply......
    """
    TYPE: ClassVar[str] = "cage_folder"
    cage_path: Path = MISSING
    dst_cage_path: Path = MISSING
    deform_cycle: int = 40  # how many frames will the deform cycle

def cage_folder_cages(args: CageFolderCages) -> CageAnimation:
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

@dataclass
class CageFolderAnimationCages:
    """
    variant of cage_folder,
    loads cage_path as src_cage,
    glob cage_path.parent / dst_cage_prefix for dst_cages, sort by .stem with prefix removed and rest treated as index
    eg: src_cage: a/b.ply prefix: c.ply, will glob a/c0.ply a/c1.ply, a/c2.ply .......
    """
    TYPE: ClassVar[str] = "cage_folder_animation"
    cage_path: Path = MISSING
    dst_cage_prefix: Path = MISSING
    interpolate: int = 0

def cage_folder_animation_cages(args: CageFolderAnimationCages) -> CageAnimation:
    src_cage: trimesh.Trimesh = trimesh.load_mesh(args.cage_path)
    log.info("reading src_cage {}", args.cage_path)

    glob_expr = f"{args.dst_cage_prefix.stem}*{args.dst_cage_prefix.suffix}"
    log.info("globbing {} / {}", args.cage_path.parent, glob_expr)
    dst_cages = list(args.cage_path.parent.glob(glob_expr))
    assert dst_cages, "must find at least one dst_cage!"
    dst_cages = sorted(dst_cages, key=lambda p: int(p.stem[len(args.dst_cage_prefix.stem):]))
    log.info("reading dst_cages {}", dst_cages)

    dst_cages = [trimesh.load_mesh(p) for p in dst_cages]

    if args.interpolate != 0:
        assert args.interpolate % 2 == 0
        interp_cages = []
        for ia, ib in zip(dst_cages, dst_cages[1:]):
            for weight in np.linspace(0.0, 1.0, args.interpolate):
                d = ib.copy()
                d.vertices = d.vertices * weight + ia.vertices * (1 - weight)
                interp_cages.append(d)
        dst_cages = interp_cages

    yield src_cage
    while True:
        for c in dst_cages:
            yield c

def emit_interpolate_weights(cycles: int, cycle: bool = True) -> Iterator[int]:
    assert cycles % 2 == 0
    s = np.linspace(0.0, 1.0, cycles // 2)
    seq = np.concatenate([s, np.flip(s, 0)], axis=0)
    if cycle:
        return itertools.cycle(seq)
    else:
        return iter(seq)

@dataclass
class ColmapTxtCameras:
    TYPE: ClassVar[str] = "colmap_txt"
    source_path: Path = MISSING
    frame_count: int = 20

def mipnerf_cameras(args: ColmapTxtCameras) -> CameraAnimation:
    cameras_extrinsic_file = os.path.join(str(args.source_path), "images.txt")
    cameras_intrinsic_file = os.path.join(str(args.source_path), "cameras.txt")
    cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    cam_infos = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics)

    cams = []
    frame_count = args.frame_count
    for c in range(len(cam_infos)-1):
        s = cam_infos[c]
        d = cam_infos[c+1]

        cis = interpolate_cameras(frame_count, s, d)

        for c in cis:
            cams.append(c)

    return cams

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

@dataclass
class FixedCameras:
    TYPE: ClassVar[str] = "fixed"
    ckpt: Path = MISSING
    frame_count: int = 20

def fixed_cameras(args: FixedCameras) -> CameraAnimation:
    chkpt = torch.load(args.ckpt)
    return [chkpt] * args.frame_count

def build_cameras(d: Dict[str, Any]) -> CameraAnimation:
    t = d["type"]
    del d["type"]
    camera_types = {
        GSDatasetCameras.TYPE: (gs_dataset_cameras, GSDatasetCameras),
        ColmapTxtCameras.TYPE: (mipnerf_cameras, ColmapTxtCameras),
        FixedCameras.TYPE: (fixed_cameras, FixedCameras)
    }
    if t in camera_types:
        f, clazz = camera_types[t]
        d = OmegaConf.merge(OmegaConf.structured(clazz), OmegaConf.create(d))
        return f(d)
    else:
        assert False

def build_cages(d: Dict[str, Any]) -> CageAnimation:
    t = d["type"]
    del d["type"]
    camera_types = {
        CageFolderCages.TYPE: (cage_folder_cages, CageFolderCages),
        CageFolderAnimationCages.TYPE: (cage_folder_animation_cages, CageFolderAnimationCages)
    }
    if t in camera_types:
        f, clazz = camera_types[t]
        d = OmegaConf.merge(OmegaConf.structured(clazz), OmegaConf.create(d))
        return f(d)