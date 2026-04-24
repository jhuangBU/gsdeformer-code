#!/bin/env bash
set -e

project_root=$(pwd)/..
run_python="conda run -n gaussian_splatting_mesh --no-capture-output python"

function run_scene() {
  set -e
  scene=$1
  $run_python train.py \
    -s "$project_root"/data/deforming_nerf-data/"$scene" \
    -m output/"$scene" \
    --gs_type gs_flat -w \
    --test_iterations 999999 \
    --save_iterations 999999
}

cd gaussian_mesh_splatting

run_scene nerf_lego
run_scene nerf_chair
run_scene nerf_hotdog
run_scene nerf_ficus
run_scene nerf_mic

run_scene nsvf_robot
run_scene nsvf_toad

#run_scene dtu_scan105
#run_scene dtu_scan83
