#!/bin/env bash
set -e

project_root=$(pwd)/..
run_python="conda run -n deforming-nerf --no-capture-output python"

function run_nerf() {
  set -e

  scene=$1
  cage_scene=$2
  cam=${3:-""}
  focalf=${4:-""}
  near=${5:-""}
  far=${6:-""}
  [[ -n "${7}" ]] && batch_size=${7}
  [[ -n "${8}" ]] && coord_batch_size=${8}
  batch_size_arg=${batch_size:+"--batch_size $batch_size"}
  coord_batch_size_arg=${coord_batch_size:+"--coord_batch_size $coord_batch_size"}

  if [ "$cam" = "ckpt" ]; then
    cam="$project_root"/data/deforming_nerf_interpolation_cages/"$scene"_camera.pt
    cam_opts="--override_cam $cam"
    override_opts="--override_focalf $focalf --override_near $near --override_far $far"
  else
    cam_opts=""
  fi
  path_ckpt_parent="$project_root"/data/deforming_nerf/nerf_"$scene"
  path_ckpt="$path_ckpt_parent"/ckpt.npz
  path_data="$project_root"/data/nerf_synthetic/"$scene"
  path_config=configs/syn.json
  path_cages="$project_root"/data/deforming_nerf_interpolation_cages

  mkdir -p "$path_ckpt_parent"/test_renders_orig
  $run_python render_imgs_deform.py \
    "$path_ckpt" "$path_data" \
    --cage_source "$path_cages"/interpolated_"$scene"_000.ply \
    --cage_target "$path_cages"/interpolated_"$scene"_"$cage_scene".ply \
    -c "$path_config" \
    --render_orig \
    --no_vid \
    $cam_opts \
    $override_opts \
    --coord_type MVC \
    $batch_size_arg \
    $coord_batch_size_arg

  mkdir -p "$path_ckpt_parent"/test_renders_deformed_MVC
  $run_python render_imgs_deform.py \
    "$path_ckpt" "$path_data" \
    --cage_source "$path_cages"/interpolated_"$scene"_000.ply \
    --cage_target "$path_cages"/interpolated_"$scene"_"$cage_scene".ply \
    -c "$path_config" \
    --no_vid \
    $cam_opts \
    $override_opts \
    --coord_type MVC \
    $batch_size_arg \
    $coord_batch_size_arg

  results_path="$project_root"/existing_methods/results-interpolate/deforming_nerf
  mkdir -p $results_path
  mv "$path_ckpt_parent"/test_renders_orig/0000.png "$results_path"/"$scene"_"$cage_scene"_og.png
  mv "$path_ckpt_parent"/test_renders_deformed_MVC/0000.png "$results_path"/"$scene"_"$cage_scene"_deformed.png
}

cd deforming_nerf/opt
for i in {000..004}; do
    run_nerf lego "$i" ckpt 0.55 0.0 2.5 15000 100000
done

for i in {000..004}; do
    run_nerf mic "$i" ckpt 0.65 0.0 2.5 10000 100000
done