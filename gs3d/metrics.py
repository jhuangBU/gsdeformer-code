#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
import re
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser


def get_mask_filename(image_name):
    """Map image filename (e.g., 'r_0.png') to mask filename (e.g., 'mask_0000.png')."""
    base = os.path.splitext(image_name)[0]
    match = re.search(r'(\d+)', base)
    if match:
        index = int(match.group(1))
        return f"mask_{index:04d}.png"
    return None


def readImages(renders_dir, gt_dir, background_color=None, mask_dir=None):
    renders = []
    gts = []
    masks = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)

        # Composite RGBA images on background if specified
        if background_color is not None:
            bg_rgb = (255, 255, 255) if background_color == 'white' else (0, 0, 0)

            if render.mode == 'RGBA':
                render_bg = Image.new('RGB', render.size, bg_rgb)
                render_bg.paste(render, mask=render.split()[3])
                render = render_bg

            if gt.mode == 'RGBA':
                gt_bg = Image.new('RGB', gt.size, bg_rgb)
                gt_bg.paste(gt, mask=gt.split()[3])
                gt = gt_bg

        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)

        # Load mask if mask_dir is provided
        if mask_dir is not None:
            mask_fname = get_mask_filename(fname)
            assert mask_fname is not None, f"Could not parse index from filename {fname}"
            mask_path = Path(mask_dir) / mask_fname
            assert mask_path.exists(), f"Mask not found for {fname} at {mask_path}"
            mask = Image.open(mask_path).convert('L')
            mask_tensor = tf.to_tensor(mask).unsqueeze(0).cuda()  # [1, 1, H, W]
            # Binarize: white (>0.5) = 1 (keep), black (<=0.5) = 0 (ignore)
            mask_tensor = (mask_tensor > 0.5).float()
            # Expand to 3 channels to match image shape [1, 3, H, W]
            mask_tensor = mask_tensor.expand(-1, 3, -1, -1)
            masks.append(mask_tensor)
        else:
            masks.append(None)

    return renders, gts, masks, image_names

def evaluate(model_paths, background_color=None, mask_dir=None):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    if background_color is not None:
        print(f"Compositing RGBA images on {background_color} background before evaluation")

    if mask_dir is not None:
        print(f"Using masks from {mask_dir} (white=keep, black=ignore)")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                renders, gts, masks, image_names = readImages(renders_dir, gt_dir, background_color, mask_dir)

                ssims = []
                psnrs = []
                lpipss = []

                bg = 1.0 if background_color == 'white' else 0.0
                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    mask = masks[idx]
                    ssims.append(ssim(renders[idx], gts[idx], mask=mask, background=bg))
                    psnrs.append(psnr(renders[idx], gts[idx], mask=mask))
                    lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg', mask=mask, background=bg))

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("")

                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss).mean().item()})
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except:
            print("Unable to compute metrics for model", scene_dir)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--background', '-b', type=str, default=None, choices=[None, 'white', 'black'],
                        help="Composite RGBA images on background before evaluation (None=no composition, white=255, black=0)")
    parser.add_argument('--masks', '-mask_dir', type=str, default=None,
                        help="Directory containing mask images (mask_{index:04d}.png). White=keep, black=ignore.")
    args = parser.parse_args()
    evaluate(args.model_paths, args.background, args.masks)
