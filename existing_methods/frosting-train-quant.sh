#!/bin/env bash
set -euo pipefail

project_root=$(pwd)/..
run_python="conda run -n frosting --no-capture-output python"

function run_scene() {
  scene_name=$1
  white_background=$2

  $run_python train_full_pipeline.py \
    -s "$project_root"/data/quantitative/"$scene_name"/"$scene_name" \
    --gaussians_in_frosting 2_000_000 \
    -r "density" \
    --white_background $white_background \
    --use_occlusion_culling False \
    --export_ply False \
    --export_obj False
}

cd frosting

run_scene $1 true