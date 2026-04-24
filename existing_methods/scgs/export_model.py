import sys
from pathlib import Path
from train_gui import GUI, MiniCam
from arguments import ModelParams, OptimizationParams, PipelineParams
from argparse import ArgumentParser
from mvc import deform_mvc_to_euclidean, euclidean_to_mvc
from gaussian_renderer import quaternion_multiply
from scene import GaussianModel
from utils.general_utils import safe_state
import torch
import trimesh
from torch import nn
import os

@torch.no_grad()
def main():
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=800, help="GUI width")
    parser.add_argument('--H', type=int, default=800, help="GUI height")
    parser.add_argument('--elevation', type=float, default=0, help="default GUI camera elevation")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[5000, 6000, 7_000] + list(range(8000, 100_0001, 1000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 10_000, 20_000, 30_000, 40000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--deform-type", type=str, default='mlp')
    parser.add_argument("--time", type=float, default=0.0, help="time to export")
    parser.add_argument("--output_ply", type=Path, default=None, help="output ply path")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    if not args.model_path.endswith(args.deform_type):
        args.model_path = os.path.join(os.path.dirname(os.path.normpath(args.model_path)), os.path.basename(os.path.normpath(args.model_path)) + f'_{args.deform_type}')

    print("Optimizing " + args.model_path)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    gui = GUI(
        args=args,
        dataset=lp.extract(args), 
        opt=op.extract(args), 
        pipe=pp.extract(args),
        testing_iterations=args.test_iterations, 
        saving_iterations=args.save_iterations
    )

    print("Initializing animation...")
    gui.animation_initialize()

    print("Saving deformed model... (pressing save_deformed)")
    
    with torch.no_grad():

        # test_step() till render() inline #

        fid = torch.remainder(torch.tensor(args.time).float().cuda(), 1.)

        cur_cam = MiniCam(
            gui.cam.pose,
            gui.W,
            gui.H,
            gui.cam.fovy,
            gui.cam.fovx,
            gui.cam.near,
            gui.cam.far,
            fid=fid
        )
        fid = cur_cam.fid

        node_trans_bias = gui.animation_trans_bias
        time_input = gui.deform.deform.expand_time(fid)
        d_values = gui.deform.step(
            gui.gaussians.get_xyz.detach(),
            time_input,
            feature=gui.gaussians.feature,
            is_training=False,
            node_trans_bias=node_trans_bias,
            motion_mask=gui.gaussians.motion_mask,
            camera_center=cur_cam.camera_center,
            animation_d_values=gui.motion_animation_d_values
        )
        gaussians = gui.gaussians
        d_xyz, d_rotation, d_scaling, d_opacity, d_color = d_values['d_xyz'], d_values['d_rotation'], d_values[
            'd_scaling'], d_values['d_opacity'], d_values['d_color']
        d_rotation_bias = d_values['d_rotation_bias'] if 'd_rotation_bias' in d_values.keys() else None

        # render() inline #

        pc = gaussians
        d_xyz = d_xyz
        d_rotation = d_rotation
        d_scaling = d_scaling
        d_opacity = d_opacity
        d_color = d_color
        d_rotation_bias = d_rotation_bias

        means3D = pc.get_xyz + d_xyz
        opacity = pc.get_opacity if d_opacity is None else pc.get_opacity + d_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = pc.get_scaling + d_scaling
        rotations = pc.get_rotation_bias(d_rotation)
        if d_rotation_bias is not None:
            rotations = quaternion_multiply(d_rotation_bias, rotations)

        sh_features = torch.cat([pc.get_features[:, :1] + d_color[:, None], pc.get_features[:, 1:]],
                                dim=1) if d_color is not None and type(d_color) is not float else pc.get_features
        shs = sh_features

        # model assembly #

        deformed_model = GaussianModel.build_from(
            gui.gaussians,
            sh_degree=gui.gaussians.max_sh_degree,
            fea_dim=gui.gaussians.fea_dim,
            with_motion_mask=gui.gaussians.with_motion_mask
        )
        deformed_model._xyz = nn.Parameter(means3D)
        deformed_model._rotation = nn.Parameter(rotations)
        scales = torch.abs(scales) # not exactly, but
        deformed_model._scaling = nn.Parameter(deformed_model.scaling_inverse_activation(scales))
        deformed_model._opacity = nn.Parameter(deformed_model.inverse_opacity_activation(opacity))
        assert torch.allclose(shs, pc.get_features) # this is our assumption for copying straight from gui.gaussians
        deformed_model._features_dc = nn.Parameter(gui.gaussians._features_dc)
        deformed_model._features_rest = nn.Parameter(gui.gaussians._features_rest)

    # # Save parameters to PyTorch file
    # params_dict = {
    #     'means3D': deformed_model.get_xyz,
    #     'shs': deformed_model.get_features,
    #     'opacities': deformed_model.get_opacity,
    #     'scales': deformed_model.get_scaling,
    #     'rotations': deformed_model.get_rotation,
    # }
    # # Generate pt file path by replacing .ply extension with .pt
    # pt_output_path = str(args.output_ply.absolute()).replace('.ply', '.pt')
    # torch.save(params_dict, pt_output_path)
    # print(f'Saved rasterizer parameters to {pt_output_path}')

    # Save the deformed model
    output_path = str(args.output_ply.absolute())
    deformed_model.save_ply(output_path, save_fea=False)
    print(f'Saved deformed Gaussian model to {output_path}')

if __name__ == "__main__":
    main()