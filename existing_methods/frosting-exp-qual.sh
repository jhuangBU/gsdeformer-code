#!/bin/env bash
set -euo pipefail

project_root=$(pwd)/..
run_python="conda run -n frosting --no-capture-output python"

cd frosting

function run_scene() {
    set -e
    scene_name=$1
    cage_scene_name=$2
    cam=$3
    threshold=${4:-2.0}
    if [ "$cam" = "ckpt" ]; then
      cam="$project_root"/data/cameras_qualitative/"$cage_scene_name"_camera_info.pt
      cam_opt="--override_cam $cam"
    else
      cam_opt="--cam_idx $cam"
    fi

    cages_path="$project_root"/data/deforming_nerf-cages-broxy-exp-qual
    data_path="$project_root"/data/deforming_nerf-data/"$scene_name"
    model_path=output/vanilla_gs/$scene_name
    model_fine_path=output/refined_frosting/$scene_name/frostingfine_3Dgs7000_densityestim02_sdfnorm02_level03_decim1000000_depthauto_quantile01_gauss2000000_frostlevel001_proposal30/15000.pt
    output_path=rendered/"$scene_name"

    $run_python visualize_results.py \
      --source_path $data_path \
      --gs_folder $model_path \
      --frosting_checkpoint $model_fine_path \
      $cam_opt \
      --output_folder "$output_path" \
      --white_background

    $run_python visualize_results.py \
      --src_cage "$cages_path"/"$cage_scene_name"_proxy.ply \
      --dst_cage "$cages_path"/"$cage_scene_name"_proxy_deformed.ply \
      --source_path $data_path \
      --gs_folder $model_path \
      --frosting_checkpoint $model_fine_path \
      $cam_opt \
      --output_folder "$output_path" \
      --white_background \
      --deformation_threshold $threshold

    results_path="$project_root"/existing_methods/results/frosting
    mkdir -p $results_path
    mv rendered/"$scene_name"/rendered_og.png "$results_path"/"$cage_scene_name"_og.png
    mv rendered/"$scene_name"/rendered.png "$results_path"/"$cage_scene_name"_deformed.png
}

#run_scene nerf_lego nerf_lego ckpt
run_scene nerf_lego nerf_lego_extreme ckpt
run_scene nerf_hotdog nerf_hotdog ckpt
run_scene nerf_ficus nerf_ficus ckpt 0.0
run_scene nerf_mic nerf_mic ckpt
#run_scene nsvf_robot nsvf_robot ckpt