#!/usr/bin/env bash
set -e

project_root=$(pwd)/..
run_python="conda run -n sugar --no-capture-output python"

cd sugar

function run_scene() {
    set -e
    scene_name=$1
    cage_scene_name=$2
    cam=$3
    if [ "$cam" = "ckpt" ]; then
      cam="$project_root"/data/deforming_nerf_interpolation_cages/"$scene_name"_camera.pt
      cam_opt="--override_cam $cam"
    else
      cam_opt="--cam_idx $cam"
    fi

    cages_path="$project_root"/data/deforming_nerf_interpolation_cages
    data_path="$project_root"/data/deforming_nerf-data/nerf_"$scene_name"
    model_path=output/nerf_"$scene_name"
    model_fine_path=output/refined/nerf_"$scene_name"/sugarfine_3Dgs7000_densityestim02_sdfnorm02_level03_decim1000000_normalconsistency01_gaussperface1
    output_path=rendered/nerf_"$scene_name"

    $run_python script_deform_mesh.py \
      --source_path "$data_path" \
      --gs_folder "$model_path" \
      --refined_sugar_folder "$model_fine_path" \
      $cam_opt \
      --output_folder "$output_path" \
      --white_bg

    $run_python script_deform_mesh.py \
      --src_cage "$cages_path"/interpolated_"$scene_name"_000.ply \
      --dst_cage "$cages_path"/interpolated_"$scene_name"_"$cage_scene_name".ply \
      --source_path "$data_path" \
      --gs_folder "$model_path" \
      --refined_sugar_folder "$model_fine_path" \
      $cam_opt \
      --output_folder "$output_path" \
      --white_bg

    results_path="$project_root"/existing_methods/results-interpolate/sugar
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