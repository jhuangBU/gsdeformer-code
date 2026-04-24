environment_setup:
  mamba env create -f environment.yml || mamba env update -f environment.yml

dataset_download_nerf_synthetic:
  kaggle datasets download -d nguyenhung1903/nerf-synthetic-dataset -p data
  unzip data/nerf-synthetic-dataset.zip -d data/nerf_synthetic
  mv data/nerf_synthetic/nerf_synthetic/* data/nerf_synthetic
  rm -rf data/nerf_synthetic/nerf_synthetic

dataset_download_dtu:
  wget https://www.dropbox.com/sh/5tam07ai8ch90pf/AAC929ngZFQUG9HOakVs_hU8a/DTU.zip -O data/DTU.zip
  unzip data/DTU.zip -d data

dataset_download_nsvf_synthetic:
  wget https://dl.fbaipublicfiles.com/nsvf/dataset/Synthetic_NSVF.zip -O data/Synthetic_NSVF.zip
  unzip data/Synthetic_NSVF.zip -d data

dataset_download_dnerf_synthetic:
    #!/bin/bash
    set -euo pipefail
    mkdir -p data/scgs-data
    wget "https://www.dropbox.com/s/0bf6fl0ye2vz3vr/data.zip?dl=1" -O data/dnerf_data.zip
    unzip data/dnerf_data.zip -d data/scgs-data
    # D-NeRF zip extracts as `data/<scene>` — flatten into data/scgs-data/<scene>
    if [[ -d data/scgs-data/data ]]; then
        mv data/scgs-data/data/* data/scgs-data/
        rmdir data/scgs-data/data
    fi
    rm data/dnerf_data.zip

dataset_model_download_deforming_nerf_ckpt_cage:
    #!/bin/bash
    folder="data/deforming_nerf"
    gdown 1WiWjSteZ6sfkoVgvSeilGgOd3BDwlCNZ --folder -O "$folder"
    for file in "$folder"/*.tar.gz; do
      tar xzf "$file" -C "$folder"
    done

dataset_preprocess_deforming_nerf_scenes:
    #!/bin/bash
    set -euo pipefail

    function process() {
        set -euo pipefail
        dst_folder=data/deforming_nerf-data/"$3"
        if [[ -e "$dst_folder" ]]; then
          echo "$3" found, skipping processing
        else
          echo "$3" not found, running processing
          if [[ $1 == "dtu" ]]; then
             mkdir "$dst_folder"
             cp -r "$2"/image "$dst_folder"/input
             cd gs3d
             python convert.py -s ../"$dst_folder"
             cd ..
          else
            python -m misc_scripts.dataset.cvt_dataset --dataset $1 --src $2 --dst $dst_folder
          fi
        fi
    }

    scenes=(chair drums ficus hotdog lego materials mic)
    for scene in "${scenes[@]}"; do
        process nerf "data/nerf_synthetic/$scene" "nerf_$scene"
    done

    # process dtu data/DTU/scan105 dtu_scan105
    # process dtu data/DTU/scan83 dtu_scan83
    process nsvf data/Synthetic_NSVF/Robot nsvf_robot
    process nsvf data/Synthetic_NSVF/Toad nsvf_toad

train_3dgs_on_deforming_nerf_scenes:
    #!/bin/bash
    set -euo pipefail

    cd gs3d

    function train_scene() {
        set -euo pipefail
        data_folder=../data/deforming_nerf-data/"$1"
        if [[ -e "$2" ]]; then
          echo "$2" found, skipping training
        else
          echo "$2" not found, running training
          python train.py -s "$data_folder" -m "$2" "${@:3}" --test_iterations 999999 --save_iterations 999999
        fi
    }

    # nerf_synthetic
    train_scene nerf_lego output/nerf_lego --white_background
    train_scene nerf_chair output/nerf_chair --white_background
    train_scene nerf_hotdog output/nerf_hotdog --white_background
    train_scene nerf_ficus output/nerf_ficus --white_background
    train_scene nerf_mic output/nerf_mic --white_background

    # NSVF
    train_scene nsvf_robot output/nsvf_robot --white_background
    train_scene nsvf_toad output/nsvf_toad --white_background

    # # DTU
    # train_scene dtu_scan105 output/dtu_scan105
    # train_scene dtu_scan83 output/dtu_scan83

infer_gsdeformer_exp_interpolate:
    #!/usr/bin/env bash
    set -eux pipefail

    function run_scene() {
        set -e

        scene="$1"
        cage_scene=$2
        algorithm=$3
        camera=$4
        if [[ $camera =~ ^[0-9]+$ ]]; then
            camera_opt="cam_idx=$camera"
        else
            camera_opt="cam_ckpt=$camera"
        fi

        source_path=data/deforming_nerf-data/nerf_$scene
        model_path=gs3d/output/nerf_$scene
        cage_path=data/deforming_nerf_interpolation_cages/interpolated_"$scene"_000.ply
        dst_cage_path=data/deforming_nerf_interpolation_cages/interpolated_"$scene"_"$cage_scene".ply
        cam_idx=$camera
        output_path=exp-qual-interpolate

        python -m gsdeformer.editorv2.cli \
          source_path=$source_path \
          model_path=$model_path \
          cage_path=$cage_path \
          dst_cage_path=$dst_cage_path \
          "$camera_opt" \
          algorithm=$algorithm \
          expname=nerf_"$scene"_"$cage_scene" \
          output_path=$output_path
    }

    for i in {000..004}; do
        run_scene lego $i all data/deforming_nerf_interpolation_cages/lego_camera.pt
    done

    for i in {000..004}; do
        run_scene mic $i all data/deforming_nerf_interpolation_cages/mic_camera.pt
    done

infer_gsdeformer_exp_highlight_interpolate:
    #!/usr/bin/env bash
    set -eux pipefail

    function run_scene() {
        set -e

        scene="$1"
        cage_scene=$2
        algorithm=$3
        camera=$4
        if [[ $camera =~ ^[0-9]+$ ]]; then
            camera_opt="cam_idx=$camera"
        else
            camera_opt="cam_ckpt=$camera"
        fi

        source_path=data/deforming_nerf-data/nerf_$scene
        model_path=gs3d/output/nerf_$scene
        cage_path=data/deforming_nerf_360_interpolation_cages/interpolated_"$scene"_000.ply
        dst_cage_path=data/deforming_nerf_360_interpolation_cages/interpolated_"$scene"_"$cage_scene".ply
        cam_idx=$camera
        output_path=exp-qual-highlight-interpolate

        python -m gsdeformer.editorv2.cli \
          source_path=$source_path \
          model_path=$model_path \
          cage_path=$cage_path \
          dst_cage_path=$dst_cage_path \
          "$camera_opt" \
          algorithm=$algorithm \
          expname=nerf_"$scene"_"$cage_scene" \
          output_path=$output_path
    }

    for i in {000..008}; do
        run_scene lego $i all data/cameras_qualitative/nerf_lego_extreme_camera_info.pt
    done

    python -m compile_qual_highlight_interpolate

infer_gsdeformer_exp_qual:
    #!/usr/bin/env bash
    set -eux pipefail

    function run_scene() {
        set -e

        scene=$1
        cage_scene=$2
        algorithm=$3
        camera=$4
        if [[ $camera =~ ^[0-9]+$ ]]; then
            camera_opt="cam_idx=$camera"
        else
            camera_opt="cam_ckpt=$camera"
        fi

        source_path=data/deforming_nerf-data/$scene
        model_path=gs3d/output/$scene
        cage_path=data/deforming_nerf-cages-broxy-exp-qual/"$cage_scene"_proxy.ply
        dst_cage_path=data/deforming_nerf-cages-broxy-exp-qual/"$cage_scene"_proxy_deformed.ply
        cam_idx=$camera
        output_path=exp-qual-results

        python -m gsdeformer.editorv2.cli \
          source_path=$source_path \
          model_path=$model_path \
          cage_path=$cage_path \
          dst_cage_path=$dst_cage_path \
          "$camera_opt" \
          algorithm=$algorithm \
          expname=$cage_scene \
          output_path=$output_path
    }

    #run_scene nerf_lego nerf_lego all data/cameras_qualitative/nerf_lego_camera_info.pt
    run_scene nerf_lego nerf_lego_extreme all data/cameras_qualitative/nerf_lego_extreme_camera_info.pt
    run_scene nerf_hotdog nerf_hotdog all data/cameras_qualitative/nerf_hotdog_camera_info.pt
    run_scene nerf_ficus nerf_ficus all data/cameras_qualitative/nerf_ficus_camera_info.pt
    run_scene nerf_mic nerf_mic all data/cameras_qualitative/nerf_mic_camera_info.pt
    #run_scene nsvf_robot nsvf_robot all data/cameras_qualitative/nsvf_robot_camera_info.pt

infer_gsdeformer_exp_qual_scgs_methods:
    #!/usr/bin/env bash

    set -eux pipefail
    export LIBGL_ALWAYS_SOFTWARE=1

    function run_scene() {
        set -e

        scene=$1
        cage_scene=$2
        camera=$3
        cage_camera=$4
        if [[ $camera =~ ^[0-9]+$ ]]; then
            camera_opt="cam_idx=$camera"
        else
            camera_opt="cam_ckpt=$camera"
            cage_camera_opt="cam_ckpt=$cage_camera"
        fi

        source_path=data/deforming_nerf-data/"$scene"
        model_path=data/scgs-output-simplified/"$cage_scene"
        cage_path=data/scgs-cages/"$cage_scene"_proxy.ply
        dst_cage_path=data/scgs-cages/"$cage_scene"_proxy_deformed.ply
        cam_idx=$camera
        output_path=exp-scgs

        python -m gsdeformer.editorv2.cli \
          source_path=$source_path \
          model_path=$model_path \
          cage_path=$cage_path \
          dst_cage_path=$dst_cage_path \
          "$camera_opt" \
          algorithm=all \
          expname=$cage_scene \
          output_path=$output_path

        python -m gsdeformer.editorv2.cli_render_cage \
          cage_path=$cage_path \
          "$cage_camera_opt" \
          output_path=$output_path/"$cage_scene"_0_all_cage.png
    }

    run_scene synth_mutant_v5 mutant data/scgs-cameras/mutant_camera_info.pt data/scgs-cameras/mutant_camera_info.pt
    run_scene synth_mutant_v5 jumpingjacks data/scgs-cameras/jumpingjacks_camera_info.pt data/scgs-cameras/jumpingjacks_camera_info.pt
    run_scene synth_mutant_v5 lego data/scgs-cameras/lego_camera_info.pt data/scgs-cameras/lego_camera_info.pt

infer_gsdeformer_exp_quant_benchmark:
    #!/usr/bin/env bash
    set -eux pipefail

    function run_experiment() {
      set -eu

      cage_type=$1
      scene_name=$2
      cage_scene_name=$3
      model_path=gs3d/output/$scene_name
      source_path=data/deforming_nerf-data/$scene_name
      if [ "$cage_type" = "deforming_nerf" ]; then
        cage_path=data/deforming_nerf/$scene_name/cages/cage.obj
        dst_cage_path=data/deforming_nerf/$scene_name/cages/cage_deformed.obj
      else
        cage_path=data/deforming_nerf-cages-broxy-exp-quant/"$cage_scene_name"_proxy.ply
        dst_cage_path=data/deforming_nerf-cages-broxy-exp-quant/"$cage_scene_name"_proxy_deformed.ply
      fi
      output_path=exp-quant-benchmark-$cage_type

      python -m gsdeformer.editorv2.cli_benchmark \
        source_path="$source_path" \
        model_path="$model_path" \
        cage_path="$cage_path" \
        dst_cage_path="$dst_cage_path" \
        expname=$cage_scene_name \
        output_path="$output_path"

      python -m gsdeformer.editorv2.cli_benchmark \
        source_path="$source_path" \
        model_path="$model_path" \
        cage_path="$cage_path" \
        dst_cage_path="$dst_cage_path" \
        output_path="$output_path" \
        expname=$cage_scene_name \
        algorithm=mean_only
    }

    run_experiment broxy nerf_lego nerf_lego
    run_experiment broxy nerf_chair nerf_chair
    run_experiment broxy nerf_ficus nerf_ficus
    run_experiment broxy nerf_hotdog nerf_hotdog

    run_experiment deforming_nerf nerf_lego nerf_lego
    run_experiment deforming_nerf nerf_chair nerf_chair
    run_experiment deforming_nerf nsvf_robot nsvf_robot
    run_experiment deforming_nerf nsvf_toad nsvf_toad

infer_gsdeformer_exp_ablation:
    #!/usr/bin/env bash
    set -eux pipefail

    function run_scene() {
        set -e

        scene=$1
        cage_scene=${2:-"$scene"}
        algorithm=$3
        camera=$4
        if [[ $camera =~ ^[0-9]+$ ]]; then
            camera_opt="cam_idx=$camera"
        else
            camera_opt="cam_ckpt=$camera"
        fi
        splitting=${5:-true}

        source_path=data/deforming_nerf-data/$scene
        model_path=gs3d/output/$scene
        cage_path=data/deforming_nerf-cages-broxy-exp-qual/"$scene"_proxy.ply
        dst_cage_path=data/deforming_nerf-cages-broxy-exp-qual/"$cage_scene"_proxy_deformed.ply
        cam_idx=$camera
        output_path=exp-qual-ablation

        python -m gsdeformer.editorv2.cli \
          source_path=$source_path \
          model_path=$model_path \
          cage_path=$cage_path \
          dst_cage_path=$dst_cage_path \
          "$camera_opt" \
          algorithm=$algorithm \
          splitting=$splitting \
          output_path=$output_path
    }

    run_scene nerf_lego nerf_lego all data/cameras_qualitative/nerf_lego_camera_info.pt
    run_scene nerf_lego nerf_lego mean_only data/cameras_qualitative/nerf_lego_camera_info.pt
    run_scene nerf_lego nerf_lego all data/cameras_qualitative/nerf_lego_camera_info.pt false

    run_scene nerf_hotdog nerf_hotdog_abl all data/cameras_qualitative/nerf_hotdog_camera_info.pt
    run_scene nerf_hotdog nerf_hotdog_abl mean_only data/cameras_qualitative/nerf_hotdog_camera_info.pt
    run_scene nerf_hotdog nerf_hotdog_abl all data/cameras_qualitative/nerf_hotdog_camera_info.pt false