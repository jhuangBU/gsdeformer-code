#!/bin/env bash
set -e

project_root=$(pwd)/..
run_python="conda run -n gaussian_splatting_mesh --no-capture-output python"

function run_scene() {
  set -e
  scene=$1
  $run_python train.py \
    -s "$project_root"/data/quantitative/"$scene"/"$scene" \
    -m output/"$scene" \
    --gs_type gs_flat -w \
    --test_iterations 999999 \
    --save_iterations 999999
}

cd gaussian_mesh_splatting

run_scene $1