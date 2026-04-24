import json
from pathlib import Path
from time import time

import dill
import torch
import svox2
import svox2.utils
import math
import argparse
import numpy as np
import os
from os import path
from util.dataset import datasets
from util import config_util

import imageio
from tqdm import tqdm

from deformation.deformation import CBD, volume_render_image_deformed
from deformation.util import load_cages, get_near_far, visualize_opt

import warnings
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser()
parser.add_argument('ckpt', type=str)

config_util.define_common_args(parser)

parser.add_argument('--n_eval', '-n', type=int, default=100000, help='images to evaluate (equal interval), at most evals every image')
parser.add_argument('--train', action='store_true', default=False, help='render train set')
parser.add_argument('--render_path',
                    action='store_true',
                    default=False,
                    help="Render path instead of test images (no metrics will be given)")
parser.add_argument('--no_vid',
                    action='store_true',
                    default=False,
                    help="Disable video generation")
parser.add_argument('--no_imsave',
                    action='store_true',
                    default=False,
                    help="Disable image saving (can still save video; MUCH faster)")
parser.add_argument('--fps',
                    type=int,
                    default=30,
                    help="FPS of video")

# Camera adjustment
parser.add_argument('--crop',
                    type=float,
                    default=1.0,
                    help="Crop (0, 1], 1.0 = full image")

# Foreground/background only
parser.add_argument('--nofg',
                    action='store_true',
                    default=False,
                    help="Do not render foreground (if using BG model)")
parser.add_argument('--nobg',
                    action='store_true',
                    default=False,
                    help="Do not render background (if using BG model)")

# Random debugging features
parser.add_argument('--blackbg',
                    action='store_true',
                    default=False,
                    help="Force a black BG (behind BG model) color; useful for debugging 'clouds'")
parser.add_argument('--ray_len',
                    action='store_true',
                    default=False,
                    help="Render the ray lengths")

# Cage setting
parser.add_argument('--cage_source', type=str, default='cage.obj', help='.obj file of source cage')
parser.add_argument('--cage_target', type=str, default='cage_deformed.obj', help='.obj file of target cage')
parser.add_argument('--cage_scale', type=float, default=None, help='Scale factor for cage vertices (e.g., 0.6667 for scene_scale compatibility)')

parser.add_argument('--coord_type', type=str, default='HC', help='type of the cage coordinate')
parser.add_argument('--coord_reso', type=int, default=128, help='resolution of the cage coordinate')
parser.add_argument('--deform_viewdirs', default=True, help='use viewdirs deformation')

# Render and interpolate setting
parser.add_argument('--render_orig', action='store_true', default=False, help="rendering original scene imgs and depths for comparison")
parser.add_argument('--cam_id', type=int, default=-1, help='fixed camera id')
parser.add_argument('--interpolate', action='store_true', default=False, help="cage interpolation")
parser.add_argument('--num_interpolate_frame', type=int, default=60, help='num of interpolated frame')
parser.add_argument('--loop', action='store_true', default=False, help="render loop animation")
parser.add_argument('--benchmark', action='store_true', default=False, help="run benchmark")
parser.add_argument('--override_cam', type=Path, default=None, help="override camera info checkpoint")
parser.add_argument('--override_focalf', type=float, default=1.0, help="correction factor for dealing with differences")
parser.add_argument('--override_near', type=float, default=0.0, help="override near for dealing with differences")
parser.add_argument('--override_far', type=float, default=2.5, help="override far for dealing with differences")
parser.add_argument('--batch_size', type=int, default=30000, help='batch_size for rendering')
parser.add_argument('--coord_batch_size', type=int, default=200000, help='batch_size for cage coordinate')
parser.add_argument('--output_folder', type=str, default=None, help='Override output folder for rendered images')
parser.add_argument('--auto_near_far', action='store_true', default=False, help='Estimate near/far from (deformed) cage per view')
parser.add_argument('--near_far_pad', type=float, default=0.05, help='Padding added to estimated near/far')
parser.add_argument('--near_far_quantile', type=float, default=0.0, help='Quantile trimming for near/far estimation (0.0 disables)')

args = parser.parse_args()
config_util.maybe_merge_config_file(args, allow_invalid=True)
device = 'cuda:0'

def _estimate_near_far_from_cage(c2w, cage_mesh, pad, quantile=0.0, min_near=1e-3):
    if cage_mesh is None or not hasattr(cage_mesh, "vertices"):
        return None

    verts = torch.as_tensor(cage_mesh.vertices, dtype=c2w.dtype, device=c2w.device)
    if verts.numel() == 0:
        return None
    ones = torch.ones((verts.shape[0], 1), dtype=verts.dtype, device=verts.device)
    verts_h = torch.cat([verts, ones], dim=1)

    w2c = torch.linalg.inv(c2w)
    
    verts_cam = verts_h @ w2c.transpose(0, 1)
    z = verts_cam[:, 2]
    z = z[torch.isfinite(z) & (z > 0)]
    if z.numel() == 0:
        return None

    if quantile and 0.0 < quantile < 0.5 and z.numel() > 8:
        z_min = torch.quantile(z, quantile)
        z_max = torch.quantile(z, 1.0 - quantile)
    else:
        z_min = z.min()
        z_max = z.max()

    z_range = (z_max - z_min).clamp_min(0.0)
    pad_total = pad + 0.01 * z_range
    near = (z_min - pad_total).clamp_min(min_near).item()
    far = (z_max + pad_total).clamp_min(near + min_near).item()

    return near, far

# Load cages
cage_source, cage_target = load_cages(args)

if not path.isfile(args.ckpt):
    args.ckpt = path.join(args.ckpt, 'ckpt.npz')

render_dir = path.join(path.dirname(args.ckpt),
            'train_renders' if args.train else 'test_renders')

if args.render_orig:
    render_dir += '_orig'
    cage_target = cage_source # only used for cage visualization
else:
    render_dir += f'_deformed_{args.coord_type}'

if args.interpolate:
    render_dir += '_interpolate'

if args.cam_id >= 0:
    render_dir += f'_cam{args.cam_id:04d}'

if args.output_folder is not None:
    render_dir = args.output_folder

dset = datasets[args.dataset_type](args.data_dir, split="test_train" if args.train else "test",
                                    **config_util.build_data_options(args))

near, far = get_near_far(dset)

grid = svox2.SparseGrid.load(args.ckpt, device=device)
grid.white_bkgd = args.white_bkgd

if grid.use_background:
    if args.nobg:
        #  grid.background_cubemap.data = grid.background_cubemap.data.cuda()
        grid.background_data.data[..., -1] = 0.0
        render_dir += '_nobg'
    if args.nofg:
        grid.density_data.data[:] = 0.0
        render_dir += '_nofg'

config_util.setup_render_opts(grid.opt, args)

if args.blackbg:
    print('Forcing black bg')
    render_dir += '_blackbg'
    grid.opt.background_brightness = 0.0

print('Writing to', render_dir)
os.makedirs(render_dir, exist_ok=True)

if not args.no_imsave:
    print('Will write out all frames as PNG (this take most of the time)')

benchmark_iters = 10
warmup_iters = 5

# NOTE: no_grad enables the fast image-level rendering kernel for cuvol backend only
# other backends will manually generate rays per frame (slow)
with torch.no_grad():
    n_images = dset.render_c2w.size(0) if args.render_path else dset.n_images
    if args.cam_id >= 0:
        n_images = args.num_interpolate_frame if args.interpolate else 1
    img_eval_interval = max(n_images // args.n_eval, 1)

    c2ws = dset.render_c2w.to(device=device) if args.render_path else dset.c2w.to(device=device)

    frames = []

    # deformation class
    pre = torch.cuda.Event(enable_timing=args.benchmark)
    post = torch.cuda.Event(enable_timing=args.benchmark)
    preprocess_times = []
    if args.benchmark:
        for _ in range(benchmark_iters):
            pre.record()
            cbd = CBD(cage_source, cage_target, coord_type=args.coord_type, res=args.coord_reso, deform_viewdirs=args.deform_viewdirs)
            cbd.ensure_cage_coordinate(args.coord_batch_size)
            post.record()
            torch.cuda.synchronize()
            preprocess_times.append(pre.elapsed_time(post))
    else:
        cbd = CBD(cage_source, cage_target, coord_type=args.coord_type, res=args.coord_reso, deform_viewdirs=args.deform_viewdirs)
        cbd.ensure_cage_coordinate(args.coord_batch_size)

    times_deform = []
    times_render = []
    if args.benchmark:
        nn_imgs = min(n_images, benchmark_iters)
    elif args.override_cam:
        nn_imgs = 1
    else:
        nn_imgs = n_images
    for img_id in tqdm(range(0, nn_imgs, img_eval_interval)):
        dset_h, dset_w = dset.get_image_size(img_id)
        im_size = dset_h * dset_w
        w = dset_w if args.crop == 1.0 else int(dset_w * args.crop)
        h = dset_h if args.crop == 1.0 else int(dset_h * args.crop)

        # choose camera
        cam_id = args.cam_id if args.cam_id >= 0 else img_id

        if args.override_cam:
            cam = torch.load(args.override_cam, pickle_module=dill)

            Rt = torch.zeros((4, 4)).to(c2ws[cam_id].device)
            Rt[:3, :3] = torch.tensor(cam.R.transpose()).to(Rt.device)
            Rt[:3, 3] = torch.tensor(cam.T).to(Rt.device)
            Rt[3, 3] = 1.0
            Rt = torch.linalg.inv(Rt)
            Rt[3,:3] = 0.0
            w, h = cam.width, cam.height

            cam = svox2.Camera(Rt,
                               dset.intrins.get('fx', cam_id) * args.override_focalf,
                               dset.intrins.get('fy', cam_id) * args.override_focalf,
                               dset.intrins.get('cx', cam_id) + (w - dset_w) * 0.5,
                               dset.intrins.get('cy', cam_id) + (h - dset_h) * 0.5,
                               w, h,
                               ndc_coeffs=dset.ndc_coeffs)

            cam.near, cam.far = args.override_near, args.override_far
        else:
            cam = svox2.Camera(c2ws[cam_id],
                               dset.intrins.get('fx', cam_id),
                               dset.intrins.get('fy', cam_id),
                               dset.intrins.get('cx', cam_id) + (w - dset_w) * 0.5,
                               dset.intrins.get('cy', cam_id) + (h - dset_h) * 0.5,
                               w, h,
                               ndc_coeffs=dset.ndc_coeffs)

            cam.near, cam.far = near, far

        # interpolate
        if args.interpolate:
            step = img_id
            if (dset.dataset_type in ['nerf', 'dtu']) and args.loop:
                step = (n_images - abs(2*img_id-n_images))
            # compute cage interpolation
            cbd.interpolate_cage(step=step, max_steps=n_images, cache=False)

        if args.auto_near_far:
            c2w_for_est = getattr(cam, "c2w", None)
            if c2w_for_est is None:
                c2w_for_est = c2ws[cam_id]
            est = _estimate_near_far_from_cage(
                c2w_for_est, cbd.cage_current, pad=args.near_far_pad, quantile=args.near_far_quantile
            )
            if est is not None:
                cam.near, cam.far = est

        # render
        im, disp, t_deform, t_render = volume_render_image_deformed(grid, cam, cbd, render_orig=args.render_orig, batch_size=args.batch_size, benchmark=args.benchmark)
        times_deform.append(t_deform)
        times_render.append(t_render)
        im.clamp_(0.0, 1.0)

        # add disparity map and cages for better visualization
        im = visualize_opt(im, disp, cam, cbd, dset)

        out_fname = f'{img_id:04d}.png'
        if (
            args.output_folder is not None
            and args.override_cam is None
            and args.cam_id < 0
            and not args.interpolate
            and hasattr(dset, 'img_files')
        ):
            out_fname = dset.img_files[cam_id]

        img_path = path.join(render_dir, out_fname)

        im = (im * 255).astype(np.uint8)
        if not args.no_imsave:
            imageio.imwrite(img_path,im)
        if not args.no_vid:
            frames.append(im)
        
        im = None

    if args.benchmark:
        preprocess_times = preprocess_times[warmup_iters:]
        times_deform = times_deform[warmup_iters:]
        times_render = times_render[warmup_iters:]
        d = {
            "preprocess_ms": preprocess_times,
            "preprocess_ms_avg": sum(preprocess_times) / len(preprocess_times),
            "deform_ms": times_deform,
            "deform_ms_avg": sum(times_deform) / len(times_deform),
            "render_ms": times_render,
            "render_ms_avg": sum(times_render) / len(times_render),
        }
        (Path(render_dir) / f"benchmark_{time()}.json").write_text(json.dumps(d, indent=4))

    if not args.no_vid and len(frames) > 1:
        vid_path = render_dir + '.gif'
        imageio.mimwrite(vid_path, frames, fps=args.fps)
        print('video saved to ', vid_path)