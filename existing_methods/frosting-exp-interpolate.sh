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
      cam="$project_root"/data/deforming_nerf_interpolation_cages/"$scene_name"_camera.pt
      cam_opt="--override_cam $cam"
    else
      cam_opt="--cam_idx $cam"
    fi

    cages_path="$project_root"/data/deforming_nerf_interpolation_cages
    data_path="$project_root"/data/deforming_nerf-data/nerf_"$scene_name"
    model_path=output/vanilla_gs/nerf_"$scene_name"
    model_fine_path=output/refined_frosting/nerf_"$scene_name"/frostingfine_3Dgs7000_densityestim02_sdfnorm02_level03_decim1000000_depthauto_quantile01_gauss2000000_frostlevel001_proposal30/15000.pt
    output_path=rendered/nerf_"$scene_name"

    $run_python visualize_results.py \
      --source_path $data_path \
      --gs_folder $model_path \
      --frosting_checkpoint $model_fine_path \
      $cam_opt \
      --output_folder "$output_path" \
      --white_background

    $run_python visualize_results.py \
      --src_cage "$cages_path"/interpolated_"$scene_name"_000.ply \
      --dst_cage "$cages_path"/interpolated_"$scene_name"_"$cage_scene_name".ply \
      --source_path $data_path \
      --gs_folder $model_path \
      --frosting_checkpoint $model_fine_path \
      $cam_opt \
      --output_folder "$output_path" \
      --white_background \
      --deformation_threshold $threshold

    results_path="$project_root"/existing_methods/results-interpolate/frosting
    mkdir -p $results_path
    mv rendered/nerf_"$scene_name"/rendered_og.png "$results_path"/"$scene_name"_"$cage_scene_name"_og.png
    mv rendered/nerf_"$scene_name"/rendered.png "$results_path"/"$scene_name"_"$cage_scene_name"_deformed.png
}

for i in {000..004}; do
    run_scene lego $i ckpt
done

for i in {000..004}; do
    run_scene mic $i ckpt
done