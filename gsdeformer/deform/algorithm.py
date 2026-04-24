import time
from pathlib import Path

import numpy as np
import open3d.geometry as ogeo
import open3d.utility as outil
import torch
import tqdm
import trimesh
import typing
from typing import Tuple

# noinspection PyUnresolvedReferences
from gsdeformer import hack_add_gs3d_to_sys_path
from scene.gaussian_model import GaussianModel
from .coords_util import mean_value_coordinates_3D
from .model_util import prune_model
from .rotation_conversions import matrix_to_quaternion, quaternion_to_matrix


def _to_pt_cloud(pts):
    return ogeo.PointCloud(points=outil.Vector3dVector(pts.numpy()))


def _to_tensor(x, device):
    return torch.from_numpy(x.astype(np.float32)).clone().to(device)


def euclidean_to_mvc(src_mesh: trimesh.Trimesh, coords: torch.Tensor, batch_size=25_000, pbar=False) -> torch.Tensor:
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

    return wjs


def deform_mvc_to_euclidean(dst_mesh: trimesh.Trimesh, mvc_coords: torch.Tensor, eu_coords: torch.Tensor, batch_size=50_000, pbar=False) -> torch.Tensor:
    cage_verts = dst_mesh.vertices
    cvt = _to_tensor(cage_verts, mvc_coords.device).unsqueeze(0)
    weights = mvc_coords

    # only deform those are in cage
    is_inside = weights.sum(-1) > 0.98  # (N,)
    weights = weights[is_inside]

    deformed = []
    it = range(0, len(weights), batch_size)
    it = tqdm.tqdm(it, desc="Deform - Cage Coord -> Euclidean") if pbar else it
    for i in it:
        w = weights[i:i + batch_size]
        d = torch.sum(w.unsqueeze(-1) * cvt, dim=1)
        deformed.append(d)
    deformed = torch.cat(deformed, 0)

    # keep the rest as-is
    ret = eu_coords.clone()
    ret[is_inside] = deformed

    return ret, is_inside


def deform_model_fast(model: GaussianModel, src_cage: Path, dst_cage: Path):
    # build transformation field

    torch.cuda.synchronize()
    start = time.time()

    # transform coordinates

    ## extract old distributions
    old_xyz = model._xyz.detach().cpu().clone()
    old_rot = model.rotation_activation(model._rotation.detach()).cpu()
    old_scaling = model.scaling_activation(model._scaling.detach()).cpu()
    old_covs = deform_cov_from_rot_scale(old_xyz, old_rot, old_scaling)

    ## build ellipsoid and points to transform
    ep_mean = old_xyz
    ep_axes, ep_scale = deform_dist_to_ellipsoid(old_covs, pdf=0.01)
    axes_pts = deform_compute_ellipsoid_pts_to_deform(ep_mean, ep_axes, ep_scale)

    # apply deformation
    torch.cuda.synchronize()
    start_deform = time.time()

    input_pts = axes_pts.reshape(-1, 3)
    cuda_input_pts = input_pts.cuda()
    mean_vals = euclidean_to_mvc(trimesh.load_mesh(src_cage), cuda_input_pts)
    axes_pts_deformed, is_inside = deform_mvc_to_euclidean(trimesh.load_mesh(dst_cage), mean_vals, cuda_input_pts)
    del mean_vals
    del cuda_input_pts
    axes_pts_deformed = axes_pts_deformed.cpu()
    is_inside = is_inside.cpu()
    axes_pts_deformed = axes_pts_deformed.reshape(-1, 4, 3)
    is_inside = is_inside.reshape(-1, 4).all(dim=1)

    torch.cuda.synchronize()
    elapsed_time = time.time() - start_deform
    print('Done to MV coord deform, time: {:.4f} sec.'.format(elapsed_time))

    # recover & apply transform
    transform = deform_deformed_pts_to_transform(axes_pts, axes_pts_deformed)
    new_mean, new_cov = deform_apply_transform(old_xyz, old_covs.to(torch.float32), transform)

    # build new distributions
    new_xyz = new_mean
    new_rot, new_scale = deform_cov_to_rot_scale(new_cov)

    # filter unreasonable values
    reasonable_xyz_change = torch.linalg.norm(new_xyz - old_xyz, dim=1) <= 10
    scaling_threshold = old_scaling.max(dim=0)[0].numpy()
    reasonable_scaling = new_scale <= torch.tensor(scaling_threshold)[np.newaxis, ...]
    reasonable_scaling = reasonable_scaling.all(dim=1)
    keep = torch.logical_and(torch.logical_and(reasonable_xyz_change, reasonable_scaling), is_inside)

    # apply mask
    # assemble model, but we prune & put in the rest for not changing / creating a new model object
    model = prune_model(model, keep)
    new_xyz = new_xyz.cuda()[keep]
    new_rot = new_rot.cuda()[keep]
    new_scale = new_scale.cuda()[keep]
    new_scale = model.scaling_inverse_activation(new_scale)
    model._xyz = new_xyz
    model._rotation = new_rot
    model._scaling = new_scale

    torch.cuda.synchronize()
    end = time.time()

    elapsed_time = end - start
    print('Deformation Complete! time: {:.4f} sec.'.format(elapsed_time))

    return model, {}


def deform_cov_from_rot_scale(old_rot: torch.Tensor, old_scaling: torch.Tensor) -> torch.Tensor:
    """
    recover covariance from rot & scale
    """
    scale = torch.zeros((old_rot.shape[0], 3, 3), dtype=torch.float32, device=old_scaling.device)
    scale[:, 0, 0] = old_scaling[:, 0]
    scale[:, 1, 1] = old_scaling[:, 1]
    scale[:, 2, 2] = old_scaling[:, 2]
    rot = quaternion_to_matrix(old_rot)[:, :3, :3]
    old_covs = torch.bmm(torch.bmm(torch.bmm(rot, scale), scale.permute(0, 2, 1)), rot.permute(0, 2, 1))
    return old_covs


def deform_dist_to_ellipsoid(old_covs: torch.Tensor, pdf: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    turn distribution into ellipsoid via setting a PDF
    """
    dim = old_covs.shape[-1]
    c1 = np.power(2 * np.pi, -dim / 2)
    c2 = torch.pow(torch.linalg.det(old_covs), -1 / 2)
    exp_out = pdf / (c1 * c2)
    assert (exp_out < 1).all(), "invalid pdf_value!"  # TODO: experimentally tested, but why???
    factor = -2 * torch.log(exp_out)
    quad = torch.linalg.inv(old_covs) / factor[:, np.newaxis, np.newaxis]
    ep_scale, ep_axes = torch.linalg.eig(quad)
    ep_scale, ep_axes = ep_scale.to(torch.float64), ep_axes.to(torch.float64)
    ep_scale = torch.sqrt(1 / ep_scale)
    return ep_axes, ep_scale


def deform_compute_ellipsoid_pts_to_deform(old_xyz:torch.Tensor, ep_axes: torch.Tensor, ep_scale: torch.Tensor) -> torch.Tensor:
    ep_mean = old_xyz
    pts = (ep_axes * ep_scale[:, np.newaxis, ...] + ep_mean[:, ..., np.newaxis]).permute(0, 2, 1)
    axes_pts = torch.concat([ep_mean[:, np.newaxis, ...], pts], dim=1)
    return axes_pts


def deform_axes_pts_to_transform(pts: torch.Tensor) -> torch.Tensor:
    """
    converts the axis representation in to affine transform that turns unit sphere into the ellipsoid it represents
    :param pts: axis repr. ed ellipsoids, shape: [N,4,3] for 3D
    :return affine transform in homo. format, shape: [N,4,4]
    """
    batch = pts.shape[0]
    dims = pts.shape[-1] + 1
    transform = torch.zeros((batch, dims, dims), device=pts.device, dtype=pts.dtype)
    for i in range(dims):
        transform[:, i, i] = 1
    transform[:, :-1, -1] = pts[:, 0, ...]
    transform[:, :-1, :-1] = (pts[:, 1:, ...] - pts[:, 0:1, ...]).permute(0,2,1)
    return transform

def deform_deformed_pts_to_transform(og_pts: torch.Tensor, d_pts: torch.Tensor) -> torch.Tensor:
    # to unit sphere transform
    ep_affine = deform_axes_pts_to_transform(og_pts.to(d_pts.device))
    tep_affine = deform_axes_pts_to_transform(d_pts)
    # ep -> tep transform
    transform = torch.bmm(tep_affine, torch.linalg.inv(ep_affine))
    return transform


def deform_apply_transform(g_mean: torch.Tensor, g_cov: torch.Tensor, transform: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    transform_t = transform[:, :-1, -1]
    transform_r = transform[:, :-1, :-1]
    tg_mean = transform_t[..., np.newaxis] + torch.bmm(transform_r, g_mean[..., np.newaxis])
    tg_mean = tg_mean[..., 0]
    tg_cov = torch.bmm(torch.bmm(transform_r, g_cov), transform_r.permute(0,2,1))
    return tg_mean, tg_cov


def deform_cov_to_rot_scale(updated_cov: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    evecs, evals, _ = torch.linalg.svd(updated_cov)
    n_axes = evecs
    n_rot = matrix_to_quaternion(n_axes)
    n_scale = torch.sqrt(evals)
    return n_rot, n_scale