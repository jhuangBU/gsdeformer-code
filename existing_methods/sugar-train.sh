#!/bin/env bash
set -euo pipefail

project_root=$(pwd)/..
run_python="conda run -n sugar --no-capture-output python"

data_folder="$project_root"/data/deforming_nerf-data

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

    CUDA_VISIBLE_DEVICES=$gpu $run_python gaussian_splatting/train.py \
      -s "$data_folder"/"$scene_name" \
      --iterations 7000 -m output/"$scene_name" \
      --test_iterations 999999 \
      --save_iterations 999999 \
      $white_opt \
      --port "$port"

    CUDA_VISIBLE_DEVICES=$gpu $run_python train.py \
      -s "$data_folder"/"$scene_name" -c output/"$scene_name"/ \
      --white_background $white_background \
      -r density \
      --export_uv_textured_mesh false \
      --export_ply false
}

cd sugar

run_scene nerf_lego 0 7771 true
run_scene nerf_chair 0 7772 true
run_scene nerf_hotdog 0 7772 true
run_scene nerf_ficus 0 7772 true
run_scene nerf_mic 0 7772 true

run_scene nsvf_robot 0 7771 true
run_scene nsvf_toad 0 7772 true

#run_scene dtu_scan105 0 7771 false
#run_scene dtu_scan83 0 7772 false
