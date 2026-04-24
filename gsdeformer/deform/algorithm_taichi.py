import contextlib
import time
from typing import Tuple

import math
import numpy as np
import torch
import trimesh
from scene.gaussian_model import GaussianModel
import taichi as ti
import taichi.math as tm
from gs3d.utils.general_utils import strip_symmetric, build_rotation
from gsdeformer.deform.algorithm import deform_cov_to_rot_scale
from gsdeformer.deform.algorithm_taichi_aux import euclidean_to_mvc, deform_mvc_to_euclidean, \
    taichi_quaterion_to_matrix, taichi_build_covs
from gsdeformer.deform.algorithm_ui import Deformer, deform_compute_ellipsoid_pts_to_deform
from gsdeformer.deform.model_util import prune_model, merge_model
from gsdeformer.deform.rotation_conversions import quaternion_to_matrix

# region profile switch

PROFILE = False

if PROFILE:
    from torch.autograd.profiler import record_function
else:
    def record_function(*args, **kwargs): return contextlib.nullcontext()

if PROFILE:
    from torch._C._autograd import ProfilerActivity
    from torch.profiler import profile
    def build_profiler():
        return profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU])
else:
    class null_profiler:

        def __init__(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def export_chrome_trace(self, *args, **kwargs):
            pass

    def build_profiler():
        return null_profiler()

# endregion

# region gaussian splitting

@ti.func
def _taichi_inplace_split_pts(
    axis: int,
    to_split_idx: int,
    extra_idx: int,
    # buffer
    axes_pts: ti.types.ndarray(dtype=tm.vec3, ndim=2),
):
    # getting points
    pt_og = axes_pts[to_split_idx, 0]
    pt_plus_axis = axes_pts[to_split_idx, axis]
    pt_neg_axis = axes_pts[to_split_idx, axis+3]

    # first half of splitting
    a_center = (pt_plus_axis + pt_og) / 2
    a_offset = a_center - pt_og
    a_plus_axis = pt_plus_axis  # old +Axis as new +Axis
    a_neg_axis = pt_og  # old Center as new -Axis

    # second half of splitting
    b_center = (pt_neg_axis + pt_og) / 2
    b_offset = b_center - pt_og
    b_plus_axis = pt_og  # old Center as new +Axis
    b_neg_axis = pt_neg_axis  # old -Axis as new -Axis

    # write second half of splitting, before overwriting old data
    for i in ti.static(range(7)):
        axes_pts[extra_idx,i] = axes_pts[to_split_idx,i] + b_offset
    axes_pts[extra_idx,0] = b_center
    axes_pts[extra_idx,axis] = b_plus_axis
    axes_pts[extra_idx,axis+3] = b_neg_axis

    # write first half of splitting
    for i in ti.static(range(7)):
        axes_pts[to_split_idx,i] = axes_pts[to_split_idx,i] + a_offset
    axes_pts[to_split_idx,0] = a_center
    axes_pts[to_split_idx,axis] = a_plus_axis
    axes_pts[to_split_idx,axis+3] = a_neg_axis

sqrt_2 = math.sqrt(2)

@ti.func
def _taichi_inplace_split_scale(
    axis: int,
    to_split_idx: int,
    extra_idx: int,
    # buffer
    scales: ti.types.ndarray(dtype=tm.vec3, ndim=1)
):
    # getting value
    v = scales[to_split_idx]
    v[axis-1] = v[axis-1] / sqrt_2
    scales[to_split_idx] = v
    scales[extra_idx] = v

@ti.func
def _taichi_inplace_split(
    axis: int,
    to_split_idx: int,
    extra_idx: int,
    # buffer
    new_axes_pt_deformed: ti.types.ndarray(dtype=tm.vec3, ndim=2)
):
    _taichi_inplace_split_pts(axis=axis, to_split_idx=to_split_idx, extra_idx=extra_idx, axes_pts=new_axes_pt_deformed)

@ti.kernel
def _taichi_split(
    to_split: ti.types.ndarray(tm.vec3, ndim=1),
    post_split_indices_m1: ti.types.ndarray(ndim=1),
    # old members
    old_axes_pts_deformed: ti.types.ndarray(tm.vec3, ndim=2),
    # new members
    new_axes_pts_deformed: ti.types.ndarray(tm.vec3, ndim=2),
    new_indices: ti.types.ndarray(ndim=1)
):
    pre_split_count = to_split.shape[0]
    for idx in ti.ndrange(pre_split_count):
        # split along x axis? y axis? z axis?
        to_split_i = to_split[idx]
        split_x, split_y, split_z = to_split_i[0], to_split_i[1],to_split_i[2]

        # where do we start storing our results?
        sidx = 0 if idx == 0 else post_split_indices_m1[idx-1]

        splat_count = 1

        # copying current splat into new location
        for i in ti.static(range(7)):
            new_axes_pts_deformed[sidx+0, i] = old_axes_pts_deformed[idx,i]

        # region runs splitting

        if split_x:
            axis = 1
            _taichi_inplace_split(axis, sidx + 0, sidx + 1, new_axes_pts_deformed)
            splat_count += 1

        if split_y:
            axis = 2
            if splat_count == 1:
                _taichi_inplace_split(axis, sidx + 0, sidx + 1, new_axes_pts_deformed)
                splat_count += 1
            elif splat_count == 2:
                _taichi_inplace_split(axis, sidx + 0, sidx + 2, new_axes_pts_deformed)
                _taichi_inplace_split(axis, sidx + 1, sidx + 3, new_axes_pts_deformed)
                splat_count *= 2

        if split_z:
            axis = 3
            if splat_count == 1:
                _taichi_inplace_split(axis, sidx + 0, sidx + 1, new_axes_pts_deformed)
                splat_count += 1
            elif splat_count == 2:
                _taichi_inplace_split(axis, sidx + 0, sidx + 2, new_axes_pts_deformed)
                _taichi_inplace_split(axis, sidx + 1, sidx + 3, new_axes_pts_deformed)
                splat_count *= 2
            elif splat_count == 4:
                _taichi_inplace_split(axis, sidx + 0, sidx + 4, new_axes_pts_deformed)
                _taichi_inplace_split(axis, sidx + 1, sidx + 5, new_axes_pts_deformed)
                _taichi_inplace_split(axis, sidx + 2, sidx + 6, new_axes_pts_deformed)
                _taichi_inplace_split(axis, sidx + 3, sidx + 7, new_axes_pts_deformed)
                splat_count *= 2

        # endregion

        # region copy other attributes

        for i in ti.ndrange(splat_count):
            new_indices[sidx+i] = idx

        # endregion

def _deform_split_gaussians(
        axes_pts_deformed: torch.Tensor,
        scaling: torch.Tensor,
        threshold_degree: float = (180 - 2) * (math.pi / 180),
        threshold_scale: float = 1e-2
):
    with record_function("prepare old data"):
        old_axes_pts_deformed = axes_pts_deformed
        old_scaling = scaling

    with record_function("compute to_split"):
        axes_a = old_axes_pts_deformed[:, 1:1+3] - old_axes_pts_deformed[:, :1]
        axes_b = old_axes_pts_deformed[:, 1+3:] - old_axes_pts_deformed[:, :1]
        axes_a_len = torch.linalg.vector_norm(axes_a, dim=-1)
        axes_b_len = torch.linalg.vector_norm(axes_b, dim=-1)
        angles_cos = (axes_a * axes_b).sum(dim=-1) / (axes_a_len * axes_b_len)
        torch.nan_to_num_(angles_cos, -1.0)
        # cross angle is below threshold (cos gt) & is long enough (scale gt)
        to_split = angles_cos > math.cos(threshold_degree)
        torch.logical_and(old_scaling >= threshold_scale, to_split, out=to_split)


    with record_function("compute post split indices"):
        # determines the post split gaussian count, split 0 times -> 1 gauss, 1 times -> 2 gauss...
        post_split_counts = torch.exp2(to_split.sum(dim=-1)).to(torch.int)
        # we store split result as splat_1 & splitted | splat_2 & splitted, instead of splats | new_splats
        post_split_indices_m1 = torch.cumsum(post_split_counts[:-1], dim=0) # result storage indices, for everyone except first splat
        post_split_count = post_split_indices_m1[-1] + post_split_counts[-1]

    # allocate new buffers

    def new_buffer(old):
        return torch.empty((post_split_count,)+old.shape[1:], device=old.device, dtype=old.dtype)

    with record_function("allocate new buffers"):
        new_axes_pts_deformed = new_buffer(axes_pts_deformed)
        new_indices = torch.empty((post_split_count,), device=axes_pts_deformed.device, dtype=torch.long)

    with record_function("run splitting kernel"):
        _taichi_split(
            to_split,
            post_split_indices_m1,
            # old members
            old_axes_pts_deformed,
            # new members
            new_axes_pts_deformed,
            new_indices
        )

    return to_split, new_axes_pts_deformed, new_indices

# endregion

# region gaussian transform

def _build_covs(rot: torch.Tensor, scaling: torch.Tensor, rot_is_matrix=False) -> torch.Tensor:
    rot_matrix = rot if rot_is_matrix else build_rotation(rot)

    l = torch.zeros((scaling.shape[0], 3, 3), dtype=torch.float, device="cuda")
    l[:, 0, 0] = scaling[:, 0]
    l[:, 1, 1] = scaling[:, 1]
    l[:, 2, 2] = scaling[:, 2]
    l = rot_matrix @ l
    L = l

    old_covs = L @ L.transpose(1, 2)
    return old_covs

def _build_inv_ep_transform(ep_axes: torch.Tensor, ep_center: torch.Tensor, ep_scale: torch.Tensor) -> torch.Tensor:
    inv_old_affine_rot = (1 / ep_scale)[:, :, None] * ep_axes.permute(0, 2, 1)
    inv_old_affine = torch.full((ep_scale.shape[0], 4, 4), 0, device=ep_scale.device, dtype=ep_scale.dtype)
    inv_old_affine[:, 3, 3] = 1.0
    inv_old_affine[:, :3, :3] = inv_old_affine_rot
    inv_old_affine[:, :3, 3] = (-inv_old_affine_rot.to(torch.float32) @ ep_center[:, :, None])[:, :, 0]
    return inv_old_affine

@ti.func
def _taichi_build_inv_ep_transform(
        ep_axes: tm.mat3,
        ep_center: tm.vec3,
        ep_scale: tm.vec3
) -> tm.mat4:
    inv_old_affine_rot = tm.mat3(
        ep_axes[0, 0] / ep_scale[0], ep_axes[1, 0] / ep_scale[0], ep_axes[2, 0] / ep_scale[0],
        ep_axes[0, 1] / ep_scale[1], ep_axes[1, 1] / ep_scale[1], ep_axes[2, 1] / ep_scale[1],
        ep_axes[0, 2] / ep_scale[2], ep_axes[1, 2] / ep_scale[2], ep_axes[2, 2] / ep_scale[2]
    )

    translation = (-inv_old_affine_rot) @ ep_center

    inv_old_affine = tm.mat4(
        inv_old_affine_rot[0, 0], inv_old_affine_rot[0, 1], inv_old_affine_rot[0, 2], translation[0],
        inv_old_affine_rot[1, 0], inv_old_affine_rot[1, 1], inv_old_affine_rot[1, 2], translation[1],
        inv_old_affine_rot[2, 0], inv_old_affine_rot[2, 1], inv_old_affine_rot[2, 2], translation[2],
        0.0, 0.0, 0.0, 1.0
    )

    return inv_old_affine

@ti.func
def _taichi_apply_transform(transform: tm.mat4, mean: tm.vec3, cov: tm.mat3) -> Tuple[tm.vec3, tm.mat3]:
    rot = tm.mat3(
        transform[0, 0], transform[0, 1], transform[0, 2],
        transform[1, 0], transform[1, 1], transform[1, 2],
        transform[2, 0], transform[2, 1], transform[2, 2]
    )
    transl = tm.vec3(transform[0, 3], transform[1, 3], transform[2, 3])

    # perform transform
    new_mean = transl + rot @ mean
    new_cov = rot @ cov @ rot.transpose()

    return new_mean, new_cov

@ti.kernel
def _taichi_transform_gaussians(
        # output
        new_means: ti.types.ndarray(dtype=tm.vec3, ndim=1),
        new_covs: ti.types.ndarray(dtype=tm.mat3, ndim=1),
        # input
        old_means: ti.types.ndarray(dtype=tm.vec3, ndim=1),
        split_axes_pts_deformed: ti.types.ndarray(dtype=tm.vec3, ndim=2),
        split_copy_indices: ti.types.ndarray(ndim=1),
        # caches
        old_covs: ti.types.ndarray(dtype=tm.mat3, ndim=1),
        old_inv_ep_transforms: ti.types.ndarray(dtype=tm.mat4, ndim=1)
):
    for idx in ti.ndrange(new_means.shape[0]):
        old_idx = split_copy_indices[idx]

        # get old_mean, old_cov
        old_mean = old_means[old_idx]
        old_cov = old_covs[old_idx]

        # ep transform
        inv_ep_transform = old_inv_ep_transforms[old_idx]

        # tep transform, estimated using 4 points instead of full 7 points as its suffice
        dpts_og = split_axes_pts_deformed[idx, 0]
        dpts_axis1 = split_axes_pts_deformed[idx, 1] - dpts_og
        dpts_axis2 = split_axes_pts_deformed[idx, 2] - dpts_og
        dpts_axis3 = split_axes_pts_deformed[idx, 3] - dpts_og
        tep_transform = tm.mat4(
            dpts_axis1[0], dpts_axis2[0], dpts_axis3[0], dpts_og[0],
            dpts_axis1[1], dpts_axis2[1], dpts_axis3[1], dpts_og[1],
            dpts_axis1[2], dpts_axis2[2], dpts_axis3[2], dpts_og[2],
            0.0, 0.0, 0.0, 1.0
        )

        # compute transform
        transform = tep_transform @ inv_ep_transform

        # apply transform
        new_mean, new_cov = _taichi_apply_transform(transform, old_mean, old_cov)

        # write results
        new_means[idx] = new_mean
        new_covs[idx] = new_cov

def _deform_transform_gaussians(
        old_mean: torch.Tensor,
        split_axes_pts_deformed: torch.Tensor,
        split_copy_indices: torch.Tensor,
        old_covs: torch.Tensor,
        old_inv_ep_transforms: torch.Tensor
):
    # init buffers
    eg = split_axes_pts_deformed
    new_mean = torch.empty((eg.shape[0],3), dtype=eg.dtype, device=eg.device)
    new_cov = torch.empty((eg.shape[0],3,3), dtype=eg.dtype, device=eg.device)
    _taichi_transform_gaussians(
        # outputs
        new_mean,
        new_cov,
        # inputs
        old_mean,
        split_axes_pts_deformed,
        split_copy_indices,
        # caches
        old_covs,
        old_inv_ep_transforms
    )
    return new_mean, new_cov

# endregion

# region main algorithm

@torch.no_grad()
def deform_model_interactive_taichi(
        model: GaussianModel,
        src_cage: trimesh.Trimesh,
        pbar:bool=False,
        return_precomp_cov:bool=True,
        compute_rot_scaling:bool=False,
        extra_filtering:bool=False,
        splitting:bool=True
) -> Deformer:
    """
    deformation algorithm, but returns a closure that turns a dst_cage into deformed GaussianModel
    """
    # transform coordinates
    t = time.time()
    with build_profiler() as prof:
        ## extract old distributions
        with record_function("extract old distributions"):
            old_xyz = model._xyz.detach().clone()
            old_rot = model.rotation_activation(model._rotation.detach())
            old_scaling = model.scaling_activation(model._scaling.detach())

        ## build ellipsoid and points to transform
        with record_function("build ellipsoid and points"):
            ep_mean = old_xyz
            ep_axes = quaternion_to_matrix(old_rot)
            ep_scale = old_scaling
            axes_pts = deform_compute_ellipsoid_pts_to_deform(ep_mean, ep_axes, ep_scale)
            del old_rot, old_scaling, ep_mean, ep_axes, ep_scale
            old_xyz = old_xyz.cpu()
            n_pts_per_ep = axes_pts.shape[1]
            n_pts_for_transform_per_ep = 4

        # apply deformation
        with record_function("pre-MVC"):
            cuda_input_pts = axes_pts.reshape(-1, 3)
            mean_vals, is_inside = euclidean_to_mvc(src_cage, cuda_input_pts, pbar=pbar)
            mean_vals = mean_vals[is_inside]
            del axes_pts

        with record_function("pre-MVC model filtering"):
            is_inside = is_inside.reshape(-1, n_pts_per_ep).all(dim=1)
            model = prune_model(model, is_inside)
            model_residual = prune_model(model, ~is_inside)

        with record_function("pre-compute cache matrices"):
            old_rot_matrix = quaternion_to_matrix(model.get_rotation)
            old_scaling = torch.clamp_min(model.get_scaling.detach(), 1e-5)  # for numerical stability
            old_covs = _build_covs(old_rot_matrix, old_scaling, rot_is_matrix=True)
            old_inv_ep_transform = _build_inv_ep_transform(old_rot_matrix, model.get_xyz, old_scaling)
            del old_rot_matrix

    prof.export_chrome_trace(f"trace_deform_{t}_prep.json")

    @torch.no_grad()
    def part_2(dst_cage: trimesh.Trimesh) -> GaussianModel:
        with build_profiler() as prof:
            with record_function("MVC"):
                axes_pts_deformed = deform_mvc_to_euclidean(dst_cage, mean_vals, pbar=pbar, batch_size=500_000)
                axes_pts_deformed = axes_pts_deformed.reshape(-1, n_pts_per_ep, 3)

            # split gaussians
            with record_function("splitting"):
                splitted, split_axes_pts_deformed, new_indices = _deform_split_gaussians(
                    axes_pts_deformed, old_scaling,
                    **({} if splitting else {"threshold_scale": float("inf")})
                )

            # recover & apply transform
            with record_function("calculate transform"):
                new_mean, new_cov = _deform_transform_gaussians(
                    old_xyz.cuda(), split_axes_pts_deformed, new_indices,
                    old_covs, old_inv_ep_transform
                )

            # build new distributions
            with record_function("compute rot scales"):
                new_xyz = new_mean
                if compute_rot_scaling:
                    # force CPU computation due to precision issues
                    new_rot, new_scale = deform_cov_to_rot_scale(new_cov.cpu())
                    new_rot = new_rot.to(new_cov.device)
                    new_scale = new_scale.to(new_cov.device)
                else:
                    new_rot = torch.zeros((new_xyz.shape[0], 4), dtype=new_xyz.dtype, device=new_xyz.device)
                    new_scale = torch.zeros((new_xyz.shape[0], 3), dtype=new_xyz.dtype, device=new_xyz.device)

            # filter unreasonable values
            with record_function("perform filtering"):
                if extra_filtering:
                    reasonable_xyz_change = torch.linalg.norm(new_xyz - old_xyz[new_indices].cuda(), dim=1) <= 10
                    scaling_diff = new_scale.max(dim=1)[0] / old_scaling[new_indices].max(dim=1)[0]
                    reasonable_scaling = scaling_diff < 10
                    keep = torch.logical_and(reasonable_xyz_change, reasonable_scaling)
                else:
                    keep = None

            # apply mask
            # assemble model, but we prune & put in the rest for not changing / creating a new model object
            with record_function("assemble model"):
                ret = GaussianModel(sh_degree=3)
                ret._xyz = new_xyz.cuda()
                ret._features_dc = model._features_dc[new_indices]
                ret._features_rest = model._features_rest[new_indices]
                ret._opacity = model._opacity[new_indices]
                ret._scaling = ret.scaling_inverse_activation(new_scale.cuda())
                ret._rotation = new_rot.cuda()
                if keep is not None:
                    ret = prune_model(ret, keep)
                    new_cov = new_cov[keep]
                ret.active_sh_degree = ret.max_sh_degree
                if return_precomp_cov:
                    new_cov = strip_symmetric(new_cov)
                    setattr(ret, "precomp_cov", new_cov)

            with record_function("merge model"):
                ret = merge_model(model_residual, ret)

        prof.export_chrome_trace(f"trace_deform_{t}_deform.json")

        return ret

    return part_2

# endregion