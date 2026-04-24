import functools
import time
from typing import Callable, Tuple

import math
import numpy as np
import torch
import trimesh
import open3d.utility as outil
import open3d.geometry as ogeo
from scipy.spatial import Delaunay
from torch.profiler import ProfilerActivity
from torch.autograd.profiler import record_function
from torch.profiler import profile

from gs3d.utils.general_utils import strip_symmetric
# noinspection PyUnresolvedReferences
from gsdeformer import hack_add_gs3d_to_sys_path
from scene.gaussian_model import GaussianModel
from gsdeformer.deform.algorithm import deform_cov_from_rot_scale, euclidean_to_mvc, deform_mvc_to_euclidean, \
    deform_apply_transform, deform_cov_to_rot_scale, deform_axes_pts_to_transform
from gsdeformer.deform.model_util import prune_model, merge_model
from gsdeformer.deform.rotation_conversions import quaternion_to_matrix

Deformer = Callable[[trimesh.Trimesh], GaussianModel]


@torch.no_grad()
def deform_in_cage_hull_only(func):
    """
    decorator function, only deforms gaussians inside convex hull of cage
    """
    @functools.wraps(func)
    def _deformer(model: GaussianModel, src_cage: trimesh.Trimesh, *args, **kwargs):
        # build convex hull
        vertices = ogeo.PointCloud(outil.Vector3dVector(src_cage.vertices))
        mesh_hull, _ = vertices.compute_convex_hull()

        # split model by convex hull
        xyz = model._xyz.detach().cpu().clone()
        hull = Delaunay(np.asarray(mesh_hull.vertices))
        inside_hull = hull.find_simplex(xyz)>=0
        model_ihull = prune_model(model, inside_hull)
        model_ohull = prune_model(model, ~inside_hull)

        factory = func(model_ihull, src_cage, *args, **kwargs)

        def _factory(*args, **kwargs):
            ret = factory(*args, **kwargs)
            return merge_model(model_ohull, ret)

        return _factory

    return _deformer


@torch.no_grad()
def deform_model_mean_only_interactive(model: GaussianModel, src_cage: trimesh.Trimesh, pbar:bool=False) -> Deformer:
    """
    deform_model interactive, but naively deforms the mean
    """
    # transform coordinates

    ## extract old distributions
    old_xyz = model._xyz.detach().cpu().clone()

    # apply deformation
    input_pts = old_xyz.reshape(-1, 3)
    cuda_input_pts = input_pts.cuda()
    mean_vals = euclidean_to_mvc(src_cage, cuda_input_pts, pbar=pbar)

    @torch.no_grad()
    def part_2(dst_cage: trimesh.Trimesh) -> GaussianModel:
        axes_pts_deformed, is_inside = deform_mvc_to_euclidean(dst_cage, mean_vals, cuda_input_pts, pbar=pbar)
        axes_pts_deformed = axes_pts_deformed.cpu()
        is_inside = is_inside.cpu()

        # build new distributions
        new_xyz = axes_pts_deformed

        # filter unreasonable values
        reasonable_xyz_change = torch.linalg.norm(new_xyz - old_xyz, dim=1) <= 10
        keep = torch.logical_and(reasonable_xyz_change, is_inside)

        # apply mask
        # assemble model, but we prune & put in the rest for not changing / creating a new model object
        new_model = prune_model(model, keep)
        new_xyz = new_xyz.cuda()[keep]
        new_model._xyz = new_xyz

        return new_model

    return part_2

def deform_compute_ellipsoid_pts_to_deform(old_xyz:torch.Tensor, ep_axes: torch.Tensor, ep_scale: torch.Tensor) -> torch.Tensor:
    ep_mean = old_xyz
    pts = (ep_mean[:, ..., np.newaxis] + ep_axes * ep_scale[:, np.newaxis, ...]).permute(0, 2, 1)
    pts2 = (ep_mean[:, ..., np.newaxis] - ep_axes * ep_scale[:, np.newaxis, ...]).permute(0, 2, 1)
    axes_pts = torch.concat([ep_mean[:, np.newaxis, ...], pts, pts2], dim=1)
    axes_pts = axes_pts.to(torch.float32)
    return axes_pts

def _deform_split_gaussians_by_axis(
        model: GaussianModel,
        axes_pts: torch.Tensor,
        axes_pts_deformed: torch.Tensor,
        is_inside: torch.Tensor,
        threshold: float = (180 - 5) * (math.pi / 180),
        axis: int = 1,
        debug: bool = False
):
    print(f"splitting axis {axis}, before", axes_pts.shape[0])
    # all data to split
    with record_function("prepare data to split"):
        axes_pts = axes_pts
        axes_pts_deformed = axes_pts_deformed
        is_inside = is_inside
        old_rot = model.get_rotation.detach()
        old_scaling = model.get_scaling.detach()
        old_scaling = torch.clamp_min(old_scaling, 1e-5)
        old_features_dc = model._features_dc.cuda()
        old_features_rest = model._features_rest.cuda()
        old_opacity = model._opacity.cuda()

    # determines the gaussians to split
    with record_function("determine to_split"):
        axes_a = axes_pts_deformed[:, axis] - axes_pts_deformed[:, 0]
        axes_b = axes_pts_deformed[:, axis + 3] - axes_pts_deformed[:, 0]
        axes_a_len = torch.linalg.vector_norm(axes_a, dim=-1)
        axes_b_len = torch.linalg.vector_norm(axes_b, dim=-1)
        axes_angles = torch.arccos((axes_a * axes_b).sum(dim=-1) / (axes_a_len * axes_b_len))
        to_split = (axes_angles < threshold) & (old_scaling[:,axis-1] >= 1e-2)

    # split

    def _split_axes_pts(pts, split_mask, name):
        """
        split selected part of axes_pts into two halves
        """
        split_pts = pts[split_mask]
        # first half of splitting
        split_pts_a = split_pts.clone()
        split_pts_a_center = (split_pts[:, axis] + split_pts[:, 0]) / 2
        split_pts_a_offset = split_pts_a_center - split_pts[:, 0]
        split_pts_a += split_pts_a_offset[:,None,:]
        split_pts_a[:, 0] = split_pts_a_center
        split_pts_a[:, axis] = (split_pts[:, axis])  # old +Axis as new +Axis
        split_pts_a[:, axis + 3] = split_pts[:, 0]  # old Center as new -Axis
        # second half of splitting
        split_pts_b = split_pts.clone()
        split_pts_b_center = (split_pts[:, axis+3] + split_pts[:, 0]) / 2
        split_pts_b_offset = split_pts_b_center - split_pts[:, 0]
        split_pts_b += split_pts_b_offset[:,None,:]
        split_pts_b[:, 0] = split_pts_b_center
        split_pts_b[:, axis] = split_pts[:, 0]  # old Center as new +Axis
        split_pts_b[:, axis + 3] = split_pts[:, axis + 3]  # old -Axis as new -Axis
        # assemble result

        # write axes pts splitted
        pts[split_mask] = split_pts_a
        pts = torch.cat([pts, split_pts_b])
        return pts

    ## deform pts
    with record_function("splitting pts"):
        axes_pts = _split_axes_pts(axes_pts, to_split, "axes_pts")
        axes_pts_deformed = _split_axes_pts(axes_pts_deformed, to_split, "axes_pts_deformed")
        # we just duplicate the config of the old points, although this is not correct!
        is_inside = torch.cat([is_inside, is_inside[to_split]])

    ## partial model for all states
    with record_function("copying models"):
        old_rot = torch.concat([old_rot, old_rot[to_split]])
        ts_scaling = old_scaling[to_split]
        ts_scaling[:, (axis - 1):(axis - 1) + 1] /= math.sqrt(2)
        old_scaling[to_split] = ts_scaling
        old_scaling = torch.concat([old_scaling, ts_scaling])
        old_features_dc = torch.cat([old_features_dc, old_features_dc[to_split]])
        old_features_rest = torch.concat([old_features_rest, old_features_rest[to_split]])
        old_opacity = torch.concat([old_opacity, old_opacity[to_split]])

    ## assemble models
    with record_function("assemble model"):
        ret = GaussianModel(model.max_sh_degree)
        ret._xyz = axes_pts[:,0]
        ret._features_dc = old_features_dc
        ret._features_rest = old_features_rest
        ret._opacity = old_opacity
        ret._scaling = ret.scaling_inverse_activation(old_scaling)
        ret._rotation = old_rot
        ret.active_sh_degree = model.max_sh_degree

    return ret, axes_pts, axes_pts_deformed, is_inside

def deform_split_gaussians(
        model: GaussianModel,
        axes_pts: torch.Tensor,
        axes_pts_deformed: torch.Tensor,
        is_inside: torch.Tensor,
        threshold: float = (180 - 5) * (math.pi / 180)
):
    model, axes_pts, axes_pts_deformed, is_inside = _deform_split_gaussians_by_axis(
        model, axes_pts, axes_pts_deformed, is_inside, threshold, axis=1
    )
    model, axes_pts, axes_pts_deformed, is_inside = _deform_split_gaussians_by_axis(
        model, axes_pts, axes_pts_deformed, is_inside, threshold, axis=2
    )
    model, axes_pts, axes_pts_deformed, is_inside = _deform_split_gaussians_by_axis(
        model, axes_pts, axes_pts_deformed, is_inside, threshold, axis=3
    )
    return model, axes_pts, axes_pts_deformed, is_inside

@torch.no_grad()
def deform_model_interactive(
        model: GaussianModel,
        src_cage: trimesh.Trimesh,
        pbar:bool=False,
        return_precomp_cov:bool=True,
        compute_rot_scaling:bool=False,
        extra_filtering:bool=False
) -> Deformer:
    """
    deformation algorithm, but returns a closure that turns a dst_cage into deformed GaussianModel
    """
    t = time.time()
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        # transform coordinates

        ## extract old distributions
        with record_function("extract old distributions"):
            old_xyz = model._xyz.detach().clone()
            old_rot = model.rotation_activation(model._rotation.detach())
            old_scaling = model.scaling_activation(model._scaling.detach())

        ## build ellipsoid and points to transform
        with record_function("build transform"):
            ep_mean = old_xyz
            ep_axes = quaternion_to_matrix(old_rot)
            ep_scale = old_scaling
            axes_pts = deform_compute_ellipsoid_pts_to_deform(ep_mean, ep_axes, ep_scale)
            del ep_mean, ep_axes, ep_scale
            n_pts_per_ep = axes_pts.shape[1]
            n_pts_for_transform_per_ep = 4

        # apply deformation
        with record_function("apply deformation"):
            input_pts = axes_pts.reshape(-1, 3)
            cuda_input_pts = input_pts
            del input_pts
            mean_vals = euclidean_to_mvc(src_cage, cuda_input_pts, pbar=pbar)

        torch.cuda.empty_cache()

    prof.export_chrome_trace(f"trace_deform_{t}_prep.json")

    @torch.no_grad()
    def part_2(dst_cage: trimesh.Trimesh) -> GaussianModel:
        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
            with record_function("apply MVC"):
                axes_pts_deformed, is_inside = deform_mvc_to_euclidean(dst_cage, mean_vals, cuda_input_pts, pbar=pbar, batch_size=25_000)
                axes_pts_deformed = axes_pts_deformed.reshape(-1, n_pts_per_ep, 3)
                is_inside = is_inside.reshape(-1, n_pts_per_ep).all(dim=1)

            # split gaussians
            with record_function("split gaussians"):
                smodel, spts, spts_deformed, is_inside = deform_split_gaussians(model, axes_pts, axes_pts_deformed, is_inside)

            # recover & apply transform
            with record_function("compute transform"):
                pts = spts[:,:n_pts_for_transform_per_ep]
                d_pts = spts_deformed[:,:n_pts_for_transform_per_ep]
                batch = pts.shape[0]
                dims = pts.shape[-1] + 1
                # ep transform
                ep_axes = quaternion_to_matrix(smodel.get_rotation)
                ep_scale = smodel.get_scaling
                inv_old_affine_rot = (1 / ep_scale)[:, :, None] * ep_axes.permute(0, 2, 1)
                inv_old_affine = torch.full((smodel.get_xyz.shape[0], 4, 4), 0, device=d_pts.device, dtype=d_pts.dtype)
                inv_old_affine[:, 3, 3] = 1.0
                inv_old_affine[:, :3, :3] = inv_old_affine_rot
                inv_old_affine[:, :3, 3] = (-inv_old_affine_rot.to(torch.float32) @ smodel.get_xyz[:, :, None])[:, :, 0]
                # tep transform
                tep_axes = d_pts[:, 1:, ...] - d_pts[:, 0:1, ...]
                tep_center, tep_rot = d_pts[:, 0, ...], tep_axes.permute(0, 2, 1)
                tep_affine = torch.zeros((batch, dims, dims), device=d_pts.device, dtype=d_pts.dtype)
                for i in range(dims):
                    tep_affine[:, i, i] = 1
                tep_affine[:, :-1, -1] = tep_center
                tep_affine[:, :-1, :-1] = tep_rot
                # ep -> tep transform
                transform = torch.bmm(tep_affine, inv_old_affine)
            with record_function("apply transform"):
                old_covs = deform_cov_from_rot_scale(smodel.get_rotation.detach(), smodel.get_scaling.detach())
                new_mean, new_cov = deform_apply_transform(smodel.get_xyz, old_covs.to(torch.float32), transform)

            # build new distributions
            with record_function("build distribution"):
                new_xyz = new_mean
                if compute_rot_scaling:
                    # force CPU computation due to precision issues
                    new_rot, new_scale = deform_cov_to_rot_scale(new_cov.cpu())
                    new_rot = new_rot.to(new_cov.device)
                    new_scale = new_scale.to(new_cov.device)
                else:
                    new_rot = torch.zeros_like(smodel.get_rotation)
                    new_scale = torch.zeros_like(smodel.get_scaling)

            # filter unreasonable values
            with record_function("filtering"):
                if extra_filtering:
                    reasonable_xyz_change = torch.linalg.norm(new_xyz - old_xyz, dim=1) <= 10
                    scaling_threshold = old_scaling.max(dim=0)[0]
                    reasonable_scaling = new_scale <= scaling_threshold[np.newaxis, ...]
                    reasonable_scaling = reasonable_scaling.all(dim=1)
                    keep = torch.logical_and(torch.logical_and(reasonable_xyz_change, reasonable_scaling), is_inside)
                else:
                    keep = is_inside

            # apply mask
            # assemble model, but we prune & put in the rest for not changing / creating a new model object
            with record_function("assemble models"):
                ret = GaussianModel(sh_degree=3)
                ret._xyz = new_xyz.cuda()[keep]
                ret._features_dc = smodel._features_dc.cuda()[keep]
                ret._features_rest = smodel._features_rest.cuda()[keep]
                ret._opacity = smodel._opacity.cuda()[keep]
                ret._scaling = ret.scaling_inverse_activation(new_scale.cuda()[keep])
                ret._rotation = new_rot.cuda()[keep]
                ret.active_sh_degree = ret.max_sh_degree
                if return_precomp_cov:
                    new_cov = strip_symmetric(new_cov)
                    setattr(ret, "precomp_cov", new_cov)

        prof.export_chrome_trace(f"trace_deform_{t}_deform.json")

        return ret

    return part_2

def build_deformer(algorithm, model, src_cage, **kwargs):
    if algorithm == "all":
        from gsdeformer.deform.algorithm_taichi import deform_model_interactive_taichi
        deformer = deform_model_interactive_taichi
    elif algorithm == "mean_only":
        deformer = deform_model_mean_only_interactive
    else:
        assert False, f"unrecognized algorithm {algorithm}"
    deformer = deform_in_cage_hull_only(deformer)
    deformer = deformer(model, src_cage, **kwargs)
    return deformer
