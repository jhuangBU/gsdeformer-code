#!/usr/bin/env bash
set -e

project_root=$(pwd)/..
run_python="conda run -n sugar --no-capture-output python"

cd sugar

function run_scene() {
    set -e

    cage_type=$1
    scene_name=$2
    cage_scene_name=$3
    cam_idx=$4

    if [ "$cage_type" = "deforming_nerf" ]; then
      cages_path="$project_root"/data/deforming_nerf/$scene_name/cages
      src_cage="$cages_path"/cage.obj
      dst_cage="$cages_path"/cage_deformed.obj
    else
      cages_path="$project_root"/data/deforming_nerf-cages-broxy-exp-quant
      src_cage="$cages_path"/"$cage_scene_name"_proxy.ply
      dst_cage="$cages_path"/"$cage_scene_name"_proxy_deformed.ply
    fi

    data_path="$project_root"/data/deforming_nerf-data/"$scene_name"
    model_path=output/"$scene_name"
    model_fine_path=output/refined/"$scene_name"/sugarfine_3Dgs7000_densityestim02_sdfnorm02_level03_decim1000000_normalconsistency01_gaussperface1
    output_path=rendered/"$scene_name"_"$cam_idx"

    $run_python script_deform_mesh.py \
      --src_cage $src_cage \
      --dst_cage $dst_cage \
      --source_path "$data_path" \
      --gs_folder "$model_path" \
      --refined_sugar_folder "$model_fine_path" \
      --cam_idx "$cam_idx" \
      --output_folder "$output_path" \
      --benchmark

  results_path="$project_root"/existing_methods/results/sugar
  mkdir -p $results_path
  mv rendered/"$scene_name"_"$cam_idx"/benchmark.json "$results_path"/benchmark_"$cage_scene_name"_"$cage_type".json
  mv rendered/"$scene_name"_"$cam_idx"/rendered.png "$results_path"/benchmark_"$cage_scene_name"_"$cage_type"_render.png
}

run_scene broxy nerf_lego nerf_lego 160
run_scene broxy nerf_chair nerf_chair 140
run_scene broxy nerf_ficus nerf_ficus 100
run_scene broxy nerf_hotdog nerf_hotdog 100

run_scene deforming_nerf nerf_lego nerf_lego 160
run_scene deforming_nerf nerf_chair nerf_chair 140
run_scene deforming_nerf nsvf_robot nsvf_robot 195
run_scene deforming_nerf nsvf_toad nsvf_toad 50

#run_scene dtu_scan105 50
#run_scene dtu_scan83 50