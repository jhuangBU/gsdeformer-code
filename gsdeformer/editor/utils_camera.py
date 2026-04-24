from typing import Tuple

import numpy as np
import open3d
import torch

# noinspection PyUnresolvedReferences
import gsdeformer.hack_add_gs3d_to_sys_path
from scene.cameras import Camera
from scene.dataset_readers import CameraInfo
from utils.graphics_utils import fov2focal, focal2fov


O3DCameraParam = Tuple[open3d.camera.PinholeCameraIntrinsic, np.ndarray]


def cvt_camera_3dgs_to_o3d_intrin(cam: CameraInfo, height=None, width=None) -> open3d.camera.PinholeCameraIntrinsic:
    cam_width = width if width else cam.width
    cam_height = height if height else cam.height
    fx = fov2focal(cam.FovX, cam_width)
    fy = fov2focal(cam.FovY, cam_height)
    cx = cam_width / 2
    cy = cam_height / 2
    intrin = open3d.camera.PinholeCameraIntrinsic(cam_width, cam_height, fx, fy, cx, cy)
    return intrin


def cvt_camera_3dgs_to_o3d_extrin(cam: CameraInfo) -> np.ndarray:
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = cam.R.transpose()
    Rt[:3, 3] = cam.T
    Rt[3, 3] = 1.0
    return Rt

def cvt_camera_o3d_to_3dgs(cam: O3DCameraParam) -> Camera:
    intrin, Rt = cam

    fx, fy = intrin.get_focal_length()
    FovY = focal2fov(fy, intrin.height)
    FovX = focal2fov(fx, intrin.width)

    R = Rt[:3, :3]
    R = R.transpose()
    T = Rt[:3, 3]

    view = Camera(
        colmap_id=1,
        R=R,
        T=T,
        FoVx=FovX,
        FoVy=FovY,
        image=torch.zeros((0, intrin.height, intrin.width)),
        gt_alpha_mask=None,
        image_name="test",
        uid=0,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
        data_device="cuda"
    )

    return view


def cvt_camera_info_to_camera(cam: CameraInfo) -> Camera:
    i = cvt_camera_3dgs_to_o3d_intrin(cam)
    e = cvt_camera_3dgs_to_o3d_extrin(cam)
    return cvt_camera_o3d_to_3dgs((i, e))