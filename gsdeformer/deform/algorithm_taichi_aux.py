from typing import Tuple

import numpy as np
import taichi as ti
import torch
import tqdm
import trimesh
import taichi.math as tm
from taichi import math as tm

# noinspection PyUnresolvedReferences
from gsdeformer import hack_add_gs3d_to_sys_path
from .coords_util import mean_value_coordinates_3D


def _to_tensor(x, device):
    return torch.from_numpy(x.astype(np.float32)).clone().to(device)


def euclidean_to_mvc(src_mesh: trimesh.Trimesh, coords: torch.Tensor, batch_size=25_000, pbar=False) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = batch_size if batch_size else len(coords)
    faces, vertices = _to_tensor(src_mesh.faces, coords.device), _to_tensor(src_mesh.vertices, coords.device)
    faces, vertices = faces.long()[None, ...], vertices[None, ...]

    with torch.no_grad():
        wjs = []
        it = range(0, len(coords), batch_size)
        it = tqdm.tqdm(it, desc="Deform - Euclidean -> Cage Coord") if pbar else it
        for i in it:
            query = coords[i:i + batch_size][None, ...]
            wj = mean_value_coordinates_3D(query, vertices, faces)
            wjs.append(wj)
        wjs = torch.cat(wjs, 1).to(coords.device)
        wjs = wjs[0]


    # only deform those are in cage
    is_inside = wjs.sum(-1) > 0.98  # (N,)
    wjs = wjs[is_inside]

    return wjs, is_inside


def deform_mvc_to_euclidean(dst_mesh: trimesh.Trimesh, mvc_coords: torch.Tensor, batch_size=50_000, pbar=False) -> torch.Tensor:
    cage_verts = dst_mesh.vertices
    cvt = _to_tensor(cage_verts, mvc_coords.device).unsqueeze(0)
    weights = mvc_coords

    deformed = []
    it = range(0, len(weights), batch_size)
    it = tqdm.tqdm(it, desc="Deform - Cage Coord -> Euclidean") if pbar else it
    for i in it:
        w = weights[i:i + batch_size]
        d = torch.sum(w.unsqueeze(-1) * cvt, dim=1)
        deformed.append(d)
    deformed = torch.cat(deformed, 0)

    return deformed


@ti.func
def taichi_quaterion_to_matrix(qu: tm.vec4) -> tm.mat3:
    norm = tm.sqrt(qu[0] * qu[0] + qu[1] * qu[1] + qu[2] * qu[2] + qu[3] * qu[3])

    r = qu[0] / norm
    x = qu[1] / norm
    y = qu[2] / norm
    z = qu[3] / norm

    R = tm.mat3(0)
    R[0, 0] = 1 - 2 * (y*y + z*z)
    R[0, 1] = 2 * (x*y - r*z)
    R[0, 2] = 2 * (x*z + r*y)
    R[1, 0] = 2 * (x*y + r*z)
    R[1, 1] = 1 - 2 * (x*x + z*z)
    R[1, 2] = 2 * (y*z - r*x)
    R[2, 0] = 2 * (x*z - r*y)
    R[2, 1] = 2 * (y*z + r*x)
    R[2, 2] = 1 - 2 * (x*x + y*y)

    return R


@ti.func
def _build_scaling_rotation(s: tm.vec3, r: tm.mat3) -> tm.mat3:
    l = tm.mat3(
        s[0], 0, 0,
        0, s[1], 0,
        0, 0, s[2]
    )
    ret = r @ l
    return ret


@ti.func
def taichi_build_covs(s: tm.vec3, r: tm.mat3) -> tm.mat3:
    l = _build_scaling_rotation(s, r)
    ret = l @ l.transpose()
    return ret
