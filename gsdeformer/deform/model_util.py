from torch import nn

# noinspection PyUnresolvedReferences
import gsdeformer.hack_add_gs3d_to_sys_path
import torch
from scene.gaussian_model import GaussianModel


def prune_model(smodel: GaussianModel, selector: torch.Tensor) -> GaussianModel:
    """prune gaussian model & related segmentation based on selector mask"""
    new_smodel = GaussianModel(smodel.max_sh_degree)
    new_smodel.active_sh_degree = smodel.active_sh_degree
    new_smodel.max_sh_degree = smodel.max_sh_degree
    # TODO: do not replace parameter with tensor
    new_smodel._xyz = smodel._xyz[selector, ...]
    new_smodel._features_dc = smodel._features_dc[selector, ...]
    new_smodel._features_rest = smodel._features_rest[selector, ...]
    new_smodel._scaling = smodel._scaling[selector, ...]
    new_smodel._rotation = smodel._rotation[selector, ...]
    new_smodel._opacity = smodel._opacity[selector, ...]
    new_smodel.max_radii2D = smodel.max_radii2D
    new_smodel.xyz_gradient_accum = smodel.xyz_gradient_accum
    new_smodel.denom = smodel.denom
    new_smodel.optimizer = smodel.optimizer
    new_smodel.percent_dense = smodel.percent_dense
    new_smodel.spatial_lr_scale = smodel.spatial_lr_scale
    return new_smodel

def merge_model(a: GaussianModel, b: GaussianModel) -> GaussianModel:
    """merge two gaussian models into one"""
    new_smodel = GaussianModel(a.max_sh_degree)

    # misc stuff, copies over
    new_smodel.active_sh_degree = a.active_sh_degree
    new_smodel.max_sh_degree = a.max_sh_degree
    # scene content
    def append(org, new, grad=True): return nn.Parameter(torch.concat([org, new], dim=0).requires_grad_(grad))
    new_smodel._xyz = append(a._xyz.detach(), b._xyz.detach())
    new_smodel._features_dc =append(a._features_dc.detach(), b._features_dc.detach())
    new_smodel._features_rest = append(a._features_rest.detach(), b._features_rest.detach())
    new_smodel._scaling = append(a._scaling.detach(), b._scaling.detach())
    new_smodel._rotation = append(a._rotation.detach(), b._rotation.detach())
    new_smodel._opacity = append(a._opacity.detach(), b._opacity.detach())
    new_smodel.max_radii2D = append(a.max_radii2D.detach(), b.max_radii2D.detach(), grad=False)
    # training state, init-ed elsewhere
    new_smodel.xyz_gradient_accum = None
    new_smodel.denom = None
    new_smodel.optimizer = None
    new_smodel.percent_dense = None
    new_smodel.spatial_lr_scale = None

    # precomp_cov handling
    a_cov = getattr(a, "precomp_cov", None)
    b_cov = getattr(b, "precomp_cov", None)
    if a_cov is not None or b_cov is not None:
        a_cov = a_cov if a_cov is not None else a.get_covariance(1.0)
        b_cov = b_cov if b_cov is not None else b.get_covariance(1.0)
        ret_cov = torch.concat([a_cov, b_cov], dim=0)
        setattr(new_smodel, "precomp_cov", ret_cov)

    return new_smodel
