#!/bin/env bash
set -e

project_root=$(pwd)/..
run_python="conda run -n deforming-nerf --no-capture-output python"

function run_nerf() {

  cage_type=$1
  scene=$2
  cage_scene=$3
  [[ -n "${4}" ]] && batch_size=${4}
  [[ -n "${5}" ]] && coord_batch_size=${5}
  batch_size_arg=${batch_size:+"--batch_size $batch_size"}
  coord_batch_size_arg=${coord_batch_size:+"--coord_batch_size $coord_batch_size"}

  path_ckpt_parent="$project_root"/data/deforming_nerf/nerf_"$scene"
  path_ckpt="$path_ckpt_parent"/ckpt.npz
  path_data="$project_root"/data/nerf_synthetic/"$scene"
  path_config=configs/syn.json

  if [ "$cage_type" = "deforming_nerf" ]; then
    path_cages=$path_ckpt_parent/cages
    path_src_cage="$path_cages"/cage.obj
    path_dst_cage="$path_cages"/cage_deformed.obj
  else
    path_cages="$project_root"/data/deforming_nerf-cages-broxy-exp-quant
    path_src_cage="$path_cages"/nerf_"$cage_scene"_proxy.ply
    path_dst_cage="$path_cages"/nerf_"$cage_scene"_proxy_deformed.ply
  fi

  mkdir -p "$path_ckpt_parent"/test_renders_deformed_MVC
  $run_python render_imgs_deform.py \
    "$path_ckpt" "$path_data" \
    -c "$path_config" \
    --benchmark \
    --no_vid \
    --coord_type MVC \
    --cage_source $path_src_cage \
    --cage_target $path_dst_cage \
    $batch_size_arg \
    $coord_batch_size_arg

  results_path="$project_root"/existing_methods/results/deforming_nerf
  mkdir -p $results_path
  mv "$path_ckpt_parent"/test_renders_deformed_MVC/benchmark_*.json "$results_path"/benchmark_nerf_"$cage_scene"_"$cage_type".json
  rm -rf "$results_path"/nerf_"$cage_scene"_benchmark_renders
  mv "$path_ckpt_parent"/test_renders_deformed_MVC "$results_path"/benchmark_nerf_"$cage_scene"_"$cage_type"_renders
}

function run_nsvf() {

  cage_type=$1
  scene=$2
  scene_cap=$3

  path_ckpt_parent="$project_root"/data/deforming_nerf/nsvf_"$scene"
  path_ckpt="$path_ckpt_parent"/ckpt.npz
  path_data="$project_root"/data/Synthetic_NSVF/"$scene_cap"
  path_config=configs/syn_nsvf.json

  if [ "$cage_type" = "deforming_nerf" ]; then
    path_cages=$path_ckpt_parent/cages
    path_src_cage="$path_cages"/cage.obj
    path_dst_cage="$path_cages"/cage_deformed.obj
  else
    path_cages="$project_root"/data/deforming_nerf-cages-broxy-exp-quant
    path_src_cage="$path_cages"/nsvf_"$scene"_proxy.ply
    path_dst_cage="$path_cages"/nsvf_"$scene"_proxy_deformed.ply
  fi

  mkdir -p "$path_ckpt_parent"/test_renders_deformed_MVC
  $run_python render_imgs_deform.py \
    "$path_ckpt" "$path_data" \
    -c "$path_config" \
    --benchmark \
    --no_vid \
    --coord_type MVC \
    --cage_source $path_src_cage \
    --cage_target $path_dst_cage

  results_path="$project_root"/existing_methods/results/deforming_nerf
  mkdir -p $results_path
  mv "$path_ckpt_parent"/test_renders_deformed_MVC/benchmark_*.json "$results_path"/benchmark_nsvf_"$scene"_"$cage_type".json
  rm -rf "$results_path"/benchmark_nsvf_"$scene"_"$cage_type"_renders
  mv "$path_ckpt_parent"/test_renders_deformed_MVC "$results_path"/benchmark_nsvf_"$scene"_"$cage_type"_renders
}

function run_dtu() {
  scene=$1
  path_ckpt_parent="$project_root"/data/deforming_nerf/dtu_"$scene"
  path_ckpt="$path_ckpt_parent"/ckpt.npz
  path_data="$project_root"/data/DTU/"$scene"
  path_config=configs/dtu.json
  mkdir -p "$path_ckpt_parent"/test_renders_deformed_MVC
  $run_python render_imgs_deform.py \
    "$path_ckpt" "$path_data" \
    -c "$path_config" \
    --benchmark \
    --no_vid \
    --coord_type MVC
}

cd deforming_nerf/opt

run_nerf broxy lego lego
run_nerf broxy chair chair
run_nerf broxy hotdog hotdog
run_nerf broxy ficus ficus 20000

run_nerf deforming_nerf lego lego
run_nerf deforming_nerf chair chair
run_nsvf deforming_nerf robot Robot
run_nsvf deforming_nerf toad Toad

#run_dtu scan105
#run_dtu scan83