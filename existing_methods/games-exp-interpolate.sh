#!/bin/env bash
set -e

project_root=$(pwd)/..
run_python="conda run -n gaussian_splatting_mesh --no-capture-output python"

cd gaussian_mesh_splatting

function run_scene() {
  set -e

  scene_name=$1
  cage_scene_name=$2
  cam=${3:-""}
  if [ "$cam" = "ckpt" ]; then
    cam="$project_root"/data/deforming_nerf_interpolation_cages/"$scene_name"_camera.pt
    cam_opts="--override_cam $cam"
  else
    cam_opts=""
  fi
  cages_path="$project_root"/data/deforming_nerf_interpolation_cages

  $run_python -m scripts.render_deform_by_cage \
    -m output/nerf_"$scene_name" \
    -s ../../data/deforming_nerf-data/nerf_"$scene_name" \
    $cam_opts

  $run_python -m scripts.render_deform_by_cage \
    -m output/nerf_"$scene_name" \
    -s ../../data/deforming_nerf-data/nerf_"$scene_name" \
    $cam_opts \
    --src_cage "$cages_path"/interpolated_"$scene_name"_000.ply \
    --dst_cage "$cages_path"/interpolated_"$scene_name"_"$cage_scene_name".ply

  results_path="$project_root"/existing_methods/results-interpolate/games
  mkdir -p $results_path
  mv output/nerf_"$scene_name"/train/ours_30000/cage_deformed_gs_points/00000_og.png "$results_path"/"$scene_name"_"$cage_scene_name"_og.png
  mv output/nerf_"$scene_name"/train/ours_30000/cage_deformed_gs_points/00000.png "$results_path"/"$scene_name"_"$cage_scene_name"_deformed.png
}

for i in {000..004}; do
    run_scene lego $i ckpt
done

for i in {000..004}; do
    run_scene mic $i ckpt
done