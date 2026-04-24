#!/bin/env bash
set -euo pipefail

project_root=$(pwd)/..
run_python="conda run -n sugar --no-capture-output python"

function run_scene() {
    scene_name=$1
    gpu=$2
    port=$3
    white_background=$4
    if [ "$white_background" = "true" ]; then
        white_opt="--white_background"
    else
        white_opt=""
    fi

    # Stage 1: Gaussian Splatting training
    CUDA_VISIBLE_DEVICES=$gpu $run_python gaussian_splatting/train.py \
      -s "$project_root"/data/quantitative/"$scene_name"/"$scene_name" \
      --iterations 7000 -m output/"$scene_name" \
      --test_iterations 999999 \
      --save_iterations 999999 \
      $white_opt \
      --port "$port"

    # Stage 2: SuGaR training
    CUDA_VISIBLE_DEVICES=$gpu $run_python train.py \
      -s "$project_root"/data/quantitative/"$scene_name"/"$scene_name" \
      -c output/"$scene_name"/ \
      --white_background $white_background \
      -r density \
      --export_uv_textured_mesh false \
      --export_ply false
}

cd sugar

run_scene $1 0 7772 true