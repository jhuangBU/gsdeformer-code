#!/bin/env bash
set -e

project_root=$(pwd)/..
run_python="conda run -n deforming-nerf --no-capture-output python"

function run_training() {
  # modified from launch.sh

  set -e
  experiment=$1
  gpu=$2
  dataset=$3

  echo Launching experiment "$experiment"
  echo GPU "$gpu"
  echo EXTRA "${@:4}"

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
    --dataset_type $dataset \
    -t "$ckpt_dir" \
    "${@:4}" 2>&1 | tee "$NOHUP_FILE"
}

cd deforming_nerf/opt

run_training nerf_lego 0 nerf "$project_root"/data/nerf_synthetic/lego -c configs/syn.json
run_training nerf_chair 0 nerf "$project_root"/data/nerf_synthetic/chair -c configs/syn.json
run_training nerf_hotdog 0 nerf "$project_root"/data/nerf_synthetic/hotdog -c configs/syn.json
run_training nerf_ficus 0 nerf "$project_root"/data/nerf_synthetic/ficus -c configs/syn.json
run_training nerf_mic 0 nerf "$project_root"/data/nerf_synthetic/mic -c configs/syn.json

run_training nsvf_robot 0 nsvf "$project_root"/data/Synthetic_NSVF/Robot -c configs/syn_nsvf.json
run_training nsvf_toad 0 nsvf "$project_root"/data/Synthetic_NSVF/Toad -c configs/syn_nsvf.json

#run_training dtu_scan105 0 dtu "$project_root"/data/DTU/scan105 -c configs/dtu.json
#run_training dtu_scan83 0 dtu "$project_root"/data/DTU/scan83 -c configs/dtu.json