#!/bin/env bash
set -e

scenes=(quant_ficus quant_mic quant_hotdog quant_lego_90 quant_lego_135 quant_lego_180)
for scene_name in "${scenes[@]}"; do
    bash sugar-train-quant.sh $scene_name
    bash games-train-quant.sh $scene_name
    bash deforming-nerf-train-quant.sh $scene_name
    bash frosting-train-quant.sh $scene_name
done