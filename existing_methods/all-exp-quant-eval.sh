#!/bin/env bash
set -e

scenes=(quant_ficus quant_mic quant_hotdog quant_lego_90 quant_lego_135 quant_lego_180)
for scene_name in "${scenes[@]}"; do
    BATCH_SIZE=10000 COORD_BATCH_SIZE=100000 bash deforming-nerf-exp-quant.sh $scene_name
    bash frosting-exp-quant.sh $scene_name
    bash games-exp-quant.sh $scene_name
    bash sugar-exp-quant.sh $scene_name
done