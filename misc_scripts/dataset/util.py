from dataclasses import dataclass
from typing import Union, List

import numpy as np
import torch


@dataclass
class Rays:
    origins: Union[torch.Tensor, List[torch.Tensor]]
    dirs: Union[torch.Tensor, List[torch.Tensor]]
    gt: Union[torch.Tensor, List[torch.Tensor]]

    def to(self, *args, **kwargs):
        origins = self.origins.to(*args, **kwargs)
        dirs = self.dirs.to(*args, **kwargs)
        gt = self.gt.to(*args, **kwargs)
        return Rays(origins, dirs, gt)

    def __getitem__(self, key):
        origins = self.origins[key]
        dirs = self.dirs[key]
        gt = self.gt[key]
        return Rays(origins, dirs, gt)

    def __len__(self):
        return self.origins.size(0)


@dataclass
class Intrin:

    fx: Union[float, torch.Tensor]
    fy: Union[float, torch.Tensor]
    cx: Union[float, torch.Tensor]
    cy: Union[float, torch.Tensor]

    def scale(self, scaling: float):
        return Intrin(
                    self.fx * scaling,
                    self.fy * scaling,
                    self.cx * scaling,
                    self.cy * scaling
                )

    def get(self, field:str, image_id:int=0):
        val = self.__dict__[field]
        return val if isinstance(val, float) else val[image_id].item()

def similarity_from_cameras(c2w):
    """
    Get a similarity transform to normalize dataset
    from c2w (OpenCV convention) cameras

    :param c2w: (N, 4)

    :return T (4,4) , scale (float)
    """
    t = c2w[:, :3, 3]
    R = c2w[:, :3, :3]

    # (1) Rotate the world so that z+ is the up axis
    # we estimate the up axis by averaging the camera up axes
    ups = np.sum(R * np.array([0, -1.0, 0]), axis=-1)
    world_up = np.mean(ups, axis=0)
    world_up /= np.linalg.norm(world_up)

    up_camspace = np.array([0.0, -1.0, 0.0])
    c = (up_camspace * world_up).sum()
    cross = np.cross(world_up, up_camspace)
    skew = np.array([[0.0, -cross[2], cross[1]],
                     [cross[2], 0.0, -cross[0]],
                     [-cross[1], cross[0], 0.0]])
    if c > -1:
        R_align = np.eye(3) + skew + (skew @ skew) * 1 / (1+c)
    else:
        # In the unlikely case the original data has y+ up axis,
        # rotate 180-deg about x axis
        R_align = np.array([[-1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0]])


    #  R_align = np.eye(3) # DEBUG
    R = (R_align @ R)
    fwds = np.sum(R * np.array([0, 0.0, 1.0]), axis=-1)
    t = (R_align @ t[..., None])[..., 0]

    # (2) Recenter the scene using camera center rays
    # find the closest point to the origin for each camera's center ray
    nearest = t + (fwds * -t).sum(-1)[:, None] * fwds

    # median for more robustness
    translate = -np.median(nearest, axis=0)

    #  translate = -np.mean(t, axis=0)  # DEBUG

    transform = np.eye(4)
    transform[:3, 3] = translate
    transform[:3, :3] = R_align

    # (3) Rescale the scene using camera distances
    scale = 1.0 / np.median(np.linalg.norm(t + translate, axis=-1))
    return transform, scale