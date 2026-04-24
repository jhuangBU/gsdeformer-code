#!/usr/bin/env bash
set -e

project_root=$(pwd)/..
run_python="conda run -n sc-gs --no-capture-output python"

cd scgs

extra_args_at_training="--deform_type node --node_num 512 --hyper_dim 8  \
  --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame --W 800 --H 800 \
  --random_bg_color --white_background"

function to_vanilla_3dgs() {
  set -e

  scene=$1
  time=$2

  data_folder="$project_root"/data/scgs-data/"$scene"
  model_folder="$project_root"/data/scgs-output/"$scene"_node
  output_folder=$project_root/data/scgs-output-simplified/"$scene"

  rm -rf "$output_folder"
  mkdir -p "$output_folder"
  cp "$model_folder"/cfg_args "$output_folder"/cfg_args
  mkdir -p "$output_folder"/point_cloud/iteration_30000

  $run_python export_model.py \
      --source_path "$data_folder" \
      --model_path "$model_folder" \
      $extra_args_at_training \
      --time $time \
      --output_ply "$output_folder"/point_cloud/iteration_30000/point_cloud.ply
}

to_vanilla_3dgs lego 0.500
to_vanilla_3dgs jumpingjacks 0.0
to_vanilla_3dgs mutant 0.0