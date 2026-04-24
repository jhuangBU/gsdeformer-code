#!/usr/bin/env bash

set -e
export LIBGL_ALWAYS_SOFTWARE=1

project_root=$(pwd)/..
run_python="conda run -n vrgs --no-capture-output python"
run_gs_cli="conda run -n vrgs --no-capture-output ./build/gs_cli"

cd vrgs

function build_tet() {
  set -e

  scene=$1
  scale=${2:-1.075}
  alpha=${3:-0.75}

  src_ply=$project_root/data/scgs-output-simplified/"$scene"/point_cloud/iteration_30000/point_cloud.ply
  dst_folder=output/"$scene"

  mkdir -p "$dst_folder"

  $run_python remesh_ops.py \
   --input "$src_ply" \
   --output "$dst_folder"/0_tetgen_"$alpha".txt \
   --scale "$scale" \
   --alpha "$alpha"

   # use the latest version for downstream
   cp "$dst_folder"/0_tetgen_"$alpha".txt "$dst_folder"/0_tetgen.txt
}

function deform_tet() {
  set -e

  scene=$1
  cage_name=$2

  cage_folder="$project_root"/data/scgs-cages
  tet_folder=output/"$scene"

  $run_python mvc_tet.py \
    --src-cage "$cage_folder"/"$cage_name"_proxy.ply \
    --dst-cage "$cage_folder"/"$cage_name"_proxy_deformed.ply \
    --tet-mesh "$tet_folder"/0_tetgen.txt \
    --output "$tet_folder"/0_tetgen_deformed.txt
}

function deform_model() {
  set -e

  scene=$1

  simplified_folder="$project_root"/data/scgs-output-simplified/"$scene"
  tet_folder=output/"$scene"

  cp "$simplified_folder"/cfg_args "$tet_folder"/cfg_args
  rm -rf "$tet_folder"/point_cloud
  mkdir -p "$tet_folder"/point_cloud/iteration_30000

  $run_gs_cli "$simplified_folder/point_cloud/iteration_30000/point_cloud.ply" \
         "$tet_folder/0_tetgen.txt" \
         "$tet_folder/0_tetgen_deformed.txt" \
         "$tet_folder/point_cloud/iteration_30000/point_cloud.ply"
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
  model_path=existing_methods/vrgs/output/"$scene"
  output_path=existing_methods/results/vrgs

  current_dir=$(pwd)
  cd "$project_root"

  python -m gsdeformer.editorv2.cli_render_only \
      source_path="$source_path" \
      model_path="$model_path" \
      "$camera_opt" \
      white_bg=true \
      expname="$scene" \
      output_path="$output_path"

  python -m gsdeformer.editorv2.cli_render_tet \
      tet_path="$model_path"/0_tetgen.txt \
      "$camera_opt" \
      output_path="$output_path"/"$scene"_tet_og.png

  python -m gsdeformer.editorv2.cli_render_tet \
      tet_path="$model_path"/0_tetgen_deformed.txt \
      "$camera_opt" \
      output_path="$output_path"/"$scene"_tet_deformed.png

  cd "$current_dir"
}

# rm -rf output/mutant
# build_tet mutant 1.075 0.1
#deform_tet mutant mutant
#deform_model mutant
# render_model mutant synth_mutant_v5 data/scgs-cameras/mutant_camera_info.pt

rm -rf output/jumpingjacks
build_tet jumpingjacks 1.075 0.1
deform_tet jumpingjacks jumpingjacks
deform_model jumpingjacks
render_model jumpingjacks synth_mutant_v5 data/scgs-cameras/jumpingjacks_camera_info.pt

rm -rf output/lego
build_tet lego 1.075 0.1
deform_tet lego lego
deform_model lego
render_model lego synth_mutant_v5 data/scgs-cameras/lego_camera_info.pt
