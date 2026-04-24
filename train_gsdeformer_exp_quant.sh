#!/bin/bash

set -e

train_quant() {
    local scene_name="$1"
    cd gs3d
    python train.py \
        -s "../data/quantitative/${scene_name}/${scene_name}" \
        -m "output/${scene_name}" \
        --white_background \
        --test_iterations 999999 \
        --save_iterations 999999 \
        --port 7070
    cd ..
}

for scene_name in quant_ficus quant_mic quant_hotdog quant_lego_90 quant_lego_135 quant_lego_180; do
    train_quant $scene_name
done