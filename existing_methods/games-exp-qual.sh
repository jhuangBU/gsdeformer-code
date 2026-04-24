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
    cam="$project_root"/data/cameras_qualitative/"$cage_scene_name"_camera_info.pt
    cam_opts="--override_cam $cam"
  else
    cam_opts=""
  fi
  cages_path="$project_root"/data/deforming_nerf-cages-broxy-exp-qual

  $run_python -m scripts.render_deform_by_cage \
    -m output/"$scene_name" \
    -s ../../data/deforming_nerf-data/"$scene_name" \
    $cam_opts

  $run_python -m scripts.render_deform_by_cage \
    -m output/"$scene_name" \
    -s ../../data/deforming_nerf-data/"$scene_name" \
    $cam_opts \
    --src_cage "$cages_path"/"$cage_scene_name"_proxy.ply \
    --dst_cage "$cages_path"/"$cage_scene_name"_proxy_deformed.ply

  results_path="$project_root"/existing_methods/results/games
  mkdir -p $results_path
  mv output/"$scene_name"/train/ours_30000/cage_deformed_gs_points/00000_og.png "$results_path"/"$cage_scene_name"_og.png
  mv output/"$scene_name"/train/ours_30000/cage_deformed_gs_points/00000.png "$results_path"/"$cage_scene_name"_deformed.png
}

#run_scene nerf_lego nerf_lego ckpt
run_scene nerf_lego nerf_lego_extreme ckpt
run_scene nerf_hotdog nerf_hotdog ckpt
run_scene nerf_ficus nerf_ficus ckpt
run_scene nerf_mic nerf_mic ckpt
# run_scene nsvf_robot nsvf_robot ckpt