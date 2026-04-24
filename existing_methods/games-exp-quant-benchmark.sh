#!/bin/env bash
set -e

project_root=$(pwd)/..
run_python="conda run -n gaussian_splatting_mesh --no-capture-output python"

cd gaussian_mesh_splatting

function run_scene() {
  set -e

  cage_type=$1
  scene_name=$2
  cage_scene_name=$3

  if [ "$cage_type" = "deforming_nerf" ]; then
    cages_path="$project_root"/data/deforming_nerf/$scene_name/cages
    src_cage="$cages_path"/cage.obj
    dst_cage="$cages_path"/cage_deformed.obj
  else
    cages_path="$project_root"/data/deforming_nerf-cages-broxy-exp-quant
    src_cage="$cages_path"/"$cage_scene_name"_proxy.ply
    dst_cage="$cages_path"/"$cage_scene_name"_proxy_deformed.ply
  fi

  $run_python -m scripts.render_deform_by_cage \
    -m output/"$scene_name" \
    -s ../../data/deforming_nerf-data/"$scene_name" \
    --src_cage $src_cage \
    --dst_cage $dst_cage \
    --benchmark

  results_path="$project_root"/existing_methods/results/games
  mkdir -p $results_path
  mv output/$scene_name/train/ours_30000/cage_deformed_gs_points/benchmark.json "$results_path"/benchmark_"$cage_scene_name"_"$cage_type".json
  mv output/$scene_name/train/ours_30000/cage_deformed_gs_points/00000.png "$results_path"/benchmark_"$cage_scene_name"_"$cage_type"_render.png
}

run_scene broxy nerf_lego nerf_lego
run_scene broxy nerf_chair nerf_chair
run_scene broxy nerf_ficus nerf_ficus
run_scene broxy nerf_hotdog nerf_hotdog

run_scene deforming_nerf nerf_lego nerf_lego
run_scene deforming_nerf nerf_chair nerf_chair
run_scene deforming_nerf nsvf_robot nsvf_robot
run_scene deforming_nerf nsvf_toad nsvf_toad

#run_scene dtu_scan105
#run_scene dtu_scan83