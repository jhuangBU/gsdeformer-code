"""
load dataset with deforming-nerf code and export it in 3DGS COLMAP format
"""
import argparse
import json
import os
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import torch
import torchvision.utils
from loguru import logger as log

# noinspection PyUnresolvedReferences
import gsdeformer.hack_add_gs3d_to_sys_path
from gs3d.scene.dataset_readers import storePly
from gs3d.utils.graphics_utils import BasicPointCloud
from gs3d.utils.sh_utils import SH2RGB
from misc_scripts.dataset.dtu_dataset import DTUDataset
from misc_scripts.dataset.nerf_dataset import NeRFDataset
from misc_scripts.dataset.nsvf_dataset import NSVFDataset
from misc_scripts.dataset.read_write_model import rotmat2qvec, Image, Camera, write_cameras_binary, write_images_binary

DEFAULT_TRAIN_PARAMETERS = {
    'data_dir': '../data/nerf_synthetic/test', 'split': 'train', 'device': 'cuda', 'factor': 1, 'n_images': None,
    'dataset_type': 'auto', 'seq_id': 1000, 'epoch_size': 64000000, 'scene_scale': None, 'scale': None,
    'white_bkgd': True, 'hold_every': 8, 'normalize_by_bbox': False, 'data_bbox_scale': 1.2, 'cam_scale_factor': 0.95,
    'normalize_by_camera': True, 'permutation': False
}

DEFAULT_TEST_PARAMETERS ={
    'data_dir': '../data/nerf_synthetic/test', 'split': 'test', 'dataset_type': 'auto', 'seq_id': 1000,
    'epoch_size': 64000000, 'scene_scale': None, 'scale': None, 'white_bkgd': True, 'hold_every': 8,
    'normalize_by_bbox': False, 'data_bbox_scale': 1.2, 'cam_scale_factor': 0.95, 'normalize_by_camera': True,
    'permutation': False
}

DATASETS = {
    "nerf": NeRFDataset,
    "dtu": DTUDataset,
    "nsvf": NSVFDataset
}

def cvt_dataset(dset_type: str, src: Path, dst: Path):
    log.info("loading dataset from {}", src)
    dataset_type = DATASETS[dset_type]
    train = dataset_type(root=str(src), **DEFAULT_TRAIN_PARAMETERS)
    test = dataset_type(root=str(src), **DEFAULT_TEST_PARAMETERS)
    extent = try_load_extent(dset_type, src, train)

    camera = extract_intrinsic(train, test)
    image_files, images, extra_remark = extract_images_extrinsics(camera, train, test, interleave=False)
    if extent is None:
        extent = compute_extent(train, test)
    pcd = generate_init_point_cloud(extent)
    write_colmap_format(dst, camera, image_files, images, pcd, extra_remark)


def try_load_extent(dset_type: str, src: Path, dset):
    # TODO: read dset_type from dset object
    if dset_type == "nerf":
        # embedded nerf synthetic extents
        tmin = np.array([-1.3, -1.3, -1.3])
        tmax = tmin + 2.6
        extent = (tmin, tmax)
    elif dset_type == "nsvf":
        bbox = np.loadtxt(src / "bbox.txt")
        tmin, tmax = bbox[:3], bbox[3:6]
        log.debug("scaling nsvf bbox by {}", dset.scene_scale)
        tmin *= dset.scene_scale
        tmax *= dset.scene_scale
        extent = (tmin, tmax)
    else:
        log.warning("reading bbox from dataset {} is not support, trying unreliable guess from cameras!")
        extent = None
    return extent


ImageImageName = Tuple[str, torch.Tensor]


def generate_init_point_cloud(extent):
    # Since this data set has no colmap data, we start with random points
    num_pts = 100_000
    print(f"Generating random point cloud ({num_pts})...")

    # We create random points inside the bounds of the synthetic Blender scenes
    tmin, tmax = extent
    xyz = np.random.random((num_pts, 3))
    xyz = xyz * (tmax - tmin)[np.newaxis,:] + tmin[np.newaxis,:]
    shs = np.random.random((num_pts, 3)) / 255.0
    pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

    return pcd

def compute_extent(train, test):
    c2ws = np.concatenate([
        train.c2w.cpu().numpy(),
        test.c2w.cpu().numpy()
    ], axis=0)
    ts = c2ws[:,:3,3]

    tmin, tmax = np.min(ts, axis=0), np.max(ts, axis=0)

    return tmin, tmax

def extract_images_extrinsics(camera: Camera, train, test, interleave:bool=True, interleave_every=8) -> Tuple[List[ImageImageName], List[Image], Dict]:
    # interleave pose & images
    train_samples = list(zip(train.c2w.cpu().numpy(), train.gt))
    test_samples = list(zip(test.c2w.cpu().numpy(), test.gt))

    interleaved = []
    extra_remark = dict()
    if interleave:
        extra_remark["test_train_blend_method"] = "interleave"
        extra_remark["interleave_every"] = interleave_every
        while train_samples:
            if len(interleaved) % interleave_every == 0:
                interleaved.append(test_samples.pop(0))
            else:
                interleaved.append(train_samples.pop(0))

        if test_samples:
            log.warning("interleaving for llffhold={}, discarded {} test samples", interleave_every, len(test_samples))
    else:
        extra_remark["test_train_blend_method"] = "concat_train_test"
        extra_remark["train_len"] = len(train_samples)
        extra_remark["test_len"] = len(test_samples)
        interleaved = [*train_samples, *test_samples]

    image_files = []
    images = []
    for idx, (c2w, gt) in enumerate(interleaved):
        filename = f"{idx:03}.png"
        image_files.append((filename, gt))

        w2c = np.linalg.inv(c2w)
        qvec = rotmat2qvec(w2c[:3, :3])
        tvec = w2c[:3, 3]
        image = Image(
            id=idx, camera_id=camera.id, name=filename,
            qvec=qvec, tvec=tvec,
            xys=np.zeros((0, 2)), point3D_ids=np.zeros((0,))
        )
        images.append(image)

    return image_files, images, extra_remark


def extract_intrinsic(train, test) -> Camera:
    train_hw = (train.h_full, train.w_full)
    test_hw = (test.h_full, test.w_full)
    assert train_hw == test_hw
    height, width = test_hw

    train_intrins = train.intrins_full
    test_intrins = test.intrins_full
    assert test_intrins == train_intrins
    intrinsic = np.array([train_intrins.fx, train_intrins.fy, train_intrins.cx, train_intrins.cy])

    camera = Camera(id=1, model="PINHOLE", width=width, height=height, params=intrinsic)

    return camera


def write_colmap_format(
        dst: Path,
        camera: Camera, image_files: List[ImageImageName], images: List[Image],
        pcd: BasicPointCloud,
        extra_remark: Dict
):
    log.info("writing dataset to {}", dst)
    dst.mkdir(parents=True, exist_ok=True)

    sparse_folder = dst / "sparse" / "0"
    sparse_folder.mkdir(parents=True, exist_ok=True)
    write_cameras_binary({camera.id: camera}, sparse_folder / "cameras.bin")
    write_images_binary({i.id: i for i in images}, sparse_folder / "images.bin")

    img_folder = dst / "images"
    img_folder.mkdir(parents=True, exist_ok=True)
    for img_name, gt in image_files:
        gt = gt.permute(2, 0, 1)
        torchvision.utils.save_image(gt, img_folder / img_name)

    storePly(str(sparse_folder / "points3D.ply"), pcd.points, pcd.colors * 255)

    (dst / "conversion_remark.json").write_text(json.dumps(extra_remark, indent=4), encoding="UTF-8")


def main(args=None):
    dataset_type, src, dst = parse_args(args)
    cvt_dataset(dataset_type, src, dst)

def parse_args(args) -> Tuple[str, Path, Path]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="nerf")
    parser.add_argument("--src", type=Path)
    parser.add_argument("--dst", type=Path)
    args = parser.parse_args(args)
    return args.dataset, args.src, args.dst


if __name__ == '__main__':
    main()