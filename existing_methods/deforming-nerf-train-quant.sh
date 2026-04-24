#!/bin/env bash
set -e

project_root=$(pwd)/..
run_python="conda run -n deforming-nerf --no-capture-output python"

function run_training() {
  # modified from launch.sh

  set -e
  experiment=$1
  gpu=$2
  dataset_type=$3
  scene_name=$4

  echo Launching experiment "$experiment"
  echo GPU "$gpu"
  echo EXTRA "${@:5}"

  ckpt_dir=ckpt/$experiment
  if [ -d "$ckpt_dir" ]; then
    echo "$ckpt_dir already exists. skipping training"
    return
  fi

  mkdir -p "$ckpt_dir"
  NOHUP_FILE=$ckpt_dir/log
  echo CKPT "$ckpt_dir"
  echo LOGFILE "$NOHUP_FILE"

  CUDA_VISIBLE_DEVICES=$gpu $run_python \
    -u opt.py \
    --eval_every 999999 \
    --save_every 999999 \
    --dataset_type $dataset_type \
    -t "$ckpt_dir" \
    "$project_root"/data/quantitative/"$experiment"/"$experiment" \
    "${@:5}" 2>&1 | tee "$NOHUP_FILE"
}

cd deforming_nerf/opt

scene_name="${1#quant_}"
run_training "$1" 0 nerf "$scene_name" -c configs/syn.json
