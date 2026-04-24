#!/usr/bin/env bash

set -e
export LIBGL_ALWAYS_SOFTWARE=1

project_root=$(pwd)/..
run_python="conda run -n sc-gs --no-capture-output python"

cd scgs

function deform_model() {
  set -e

  scene=$1
  cage_name=$2
  time=$3

  cage_folder="$project_root"/data/scgs-cages
  data_folder="$project_root"/data/scgs-data/"$scene"
  model_folder="$project_root"/data/scgs-output/"$scene"_node
  output_folder=output/"$scene"


  rm -rf "$output_folder"
  mkdir -p "$output_folder"
  cp "$model_folder"/cfg_args "$output_folder"/cfg_args
  mkdir -p "$output_folder"/point_cloud/iteration_30000

  $run_python train_cli.py \
      --source_path "$data_folder" \
      --model_path "$model_folder" \
      --deform_type node --node_num 512 --hyper_dim 8 --is_blender --time $time \
      --src_mesh "$cage_folder"/"$cage_name"_proxy.ply \
      --dst_mesh "$cage_folder"/"$cage_name"_proxy_deformed.ply \
      --output_ply "$output_folder"/point_cloud/iteration_30000/point_cloud.ply
}

function render_model() {
  set -e

  scene=$1
  cage_scene=$2
  camera=$3

  if [[ $camera =~ ^[0-9]+$ ]]; then
      camera_opt="cam_idx=$camera"
  else
      camera_opt="cam_ckpt=$camera"
  fi

  source_path=data/deforming_nerf-data/"$cage_scene"
  model_path=existing_methods/scgs/output/"$scene"
  output_path=existing_methods/results/scgs

  current_dir=$(pwd)
  cd "$project_root"

  python -m gsdeformer.editorv2.cli_render_only \
      source_path="$source_path" \
      model_path="$model_path" \
      "$camera_opt" \
      white_bg=true \
      expname="$scene" \
      output_path="$output_path"

  python -m gsdeformer.editorv2.cli_render_scgs \
      ctrl_path="$model_path"/point_cloud/iteration_30000/point_cloud_control.ply \
      "$camera_opt" \
      output_path="$output_path"/"$scene"_scgs_og.png

  python -m gsdeformer.editorv2.cli_render_scgs \
      ctrl_path="$model_path"/point_cloud/iteration_30000/point_cloud_control_deformed.ply \
      "$camera_opt" \
      output_path="$output_path"/"$scene"_scgs_deformed.png

  cd "$current_dir"
}

#deform_model mutant mutant 0.0
#render_model mutant synth_mutant_v5 data/scgs-cameras/mutant_camera_info.pt

deform_model jumpingjacks jumpingjacks 0.0
render_model jumpingjacks synth_mutant_v5 data/scgs-cameras/jumpingjacks_camera_info.pt

deform_model lego lego 0.500
render_model lego synth_mutant_v5 data/scgs-cameras/lego_camera_info.pt
