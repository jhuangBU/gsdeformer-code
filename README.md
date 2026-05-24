# GSDeformer

main deformation code for paper [GSDeformer: Direct, Real-time and Extensible Cage-based Deformation for 3D Gaussian Splatting](https://arxiv.org/abs/2405.15491)

For cage building code dump, please check [gsdeformer-cage](https://github.com/jhuangBU/gsdeformer-cage)

## End-to-end reproduction script for the highlight image

```bash
bash reproduce_highlight_interpolate.sh
```

End-to-end script for a vanilla Ubuntu 20.04.1 LTS host with an NVIDIA GPU. Reproduces the teaser figure of the paper (without the text annotations overlaid in the published version).

**No training is required.** The script downloads pre-trained 3DGS weights for the lego scene from HuggingFace and runs inference only.

Prereqs: NVIDIA driver ≥ 510 (check `nvidia-smi`), `sudo` access for apt.

Installs system deps, miniforge, `just`, and the conda env; pulls cages/cameras and a pre-trained 3DGS lego bundle from HuggingFace (~7 GB unzipped); Outputs land in `exp-qual-highlight-interpolate/` and `stacked.png` (2560×1024).

Inference time on the reference hardware (1× NVIDIA GTX 1080, tested on a vast.ai instance): **~3m 38s** wall-clock for `just infer_gsdeformer_exp_highlight_interpolate` (the rendering step that produces `stacked.png`).

See `metadata.txt` for reference hardware, target figure, and reproduction notes.

## Requirements

### Hardware

- NVIDIA GPU, 6GB+ VRAM (inference; more for training), CUDA 11.6 compatible driver (≥ 510)

### Software

- Linux, ideally with desktop environment (check note for running code in headless mode)
- `git`
- `miniforge` for mamba
- `just`
- `xvfb` for running cli in headless environment

### Note: Headless Mode

Rendering code (`gsdeformer.editorv2.cli`, `gsdeformer.editorv2` GUI, and every `just infer_*` task) uses Open3D's Filament backend, which needs a display. On a headless server, install `xvfb` and prefix the command with `xvfb-run -a`:

```bash
sudo apt install xvfb
xvfb-run -a just infer_gsdeformer_exp_highlight_interpolate
xvfb-run -a python -m gsdeformer.editorv2.cli ...
```

## Setup Environment

Please refer to Justfile task environment_setup for environment setup:
```bash
just environment_setup
```

For existing methods in existing_methods, call all-environment-setup.sh
```bash
cd existing_methods && bash all-environment-setup.sh && cd ..
```

## Download Dataset & Model

1. Download Dataset from https://huggingface.co/datasets/jjhuangbu/gsdeformer-data, and put it as data folder in repo, after setup it should look like:

```
<repo root>/
└── data/
    ├── cameras_qualitative/
    ├── deforming_nerf/
    ├── deforming_nerf-cages-broxy-exp-qual/
    ├── deforming_nerf-cages-broxy-exp-quant/
    ├── deforming_nerf_360_interpolation_cages/
    ├── deforming_nerf_interpolation_cages/
    ├── quantitative/
    ├── scgs-cages/
    └── scgs-cameras/
```

2. download all other required datasets
```bash
just dataset_download_nerf_synthetic
just dataset_download_nsvf_synthetic
just dataset_preprocess_deforming_nerf_scenes
just dataset_model_download_deforming_nerf_ckpt_cage
```

## Scripts for Experiments

```bash

# Common setup #

# Shared 3DGS + baseline training on 7 scenes:
#   nerf_{lego,chair,hotdog,ficus,mic} + nsvf_{robot,toad}
# (used by experiments 1, 2, 4, 5)
just train_3dgs_on_deforming_nerf_scenes
cd existing_methods && bash all-exp-qual-train.sh && cd ..

# Experiment 1: Qualitative #

just infer_gsdeformer_exp_qual
cd existing_methods && bash all-exp-qual.sh && cd ..

# Outputs:
#   GSDeformer:  exp-qual-results/
#   Baselines:   existing_methods/results/{sugar,games,deforming_nerf,frosting}/<scene>_{og,deformed}.png

# Experiment 2: Qualitative Interpolation #

just infer_gsdeformer_exp_interpolate
just infer_gsdeformer_exp_highlight_interpolate
cd existing_methods && bash all-exp-interpolate.sh && cd ..

# Outputs:
#   GSDeformer (standard):  exp-qual-interpolate/
#   GSDeformer (360 highlight, lego only):  exp-qual-highlight-interpolate/
#   Baselines:  existing_methods/results-interpolate/{sugar,games,deforming_nerf,frosting}/


# Experiment 3: Quantitative Quality #

bash train_gsdeformer_exp_quant.sh
cd existing_methods && bash all-exp-quant-train.sh && cd ..

bash infer_gsdeformer_exp_quant_all.sh
cd existing_methods && bash all-exp-quant-eval.sh && cd ..
python compile_quant_quality_results.py

# Outputs:
#   GSDeformer per-scene:  gs3d/output/eval_quant_*/results.json
#   Baselines per-scene:   existing_methods/{sugar,gaussian_mesh_splatting,frosting,deforming_nerf}/output/eval_quant_*/results.json
#   Aggregated table:      compiled_results_quant_quality.csv


# Experiment 4: Quantitative Speed Benchmark #

just infer_gsdeformer_exp_quant_benchmark
cd existing_methods && bash all-exp-quant-benchmark.sh && cd ..
python compile_exp_quant.py

# Outputs:
#   GSDeformer:  exp-quant-benchmark-broxy/, exp-quant-benchmark-deforming_nerf/
#   Baselines:   existing_methods/results/{sugar,games,deforming_nerf,frosting}/benchmark_*.json
#   Aggregated:  exp_{broxy,deforming_nerf}_all_train_stats{,_avg}.csv
#                exp_{broxy,deforming_nerf}_all_deform_stats{,_avg,_pivoted}.csv


# Experiment 5: Ablation #
# Reuses gs3d/output/{nerf_lego,nerf_hotdog} from common training.

just infer_gsdeformer_exp_ablation

# Outputs:
#   GSDeformer:  exp-qual-ablation/

```

## Script for GUI

```bash
python -m gsdeformer.editorv2 \
  source_path=data/deforming_nerf-data/nerf_lego \
  model_path=gs3d/output/nerf_lego \
  cage_path=data/deforming_nerf-cages-broxy-exp-qual/nerf_lego_proxy.ply \
  white_bg=true
```

## Citation

```
@ARTICLE{11494421,
  author={Huang, Jiajun and Xu, Shuolin and Yu, Hongchuan and Lee, Tong-Yee},
  journal={IEEE Transactions on Visualization and Computer Graphics}, 
  title={GSDeformer: Direct, Real-time and Extensible Cage-based Deformation for 3D Gaussian Splatting}, 
  year={2026},
  volume={},
  number={},
  pages={1-16},
  keywords={Videos;Video equipment;Radio access networks;Regional area networks;Graphical user interfaces;MISO;Protocols;Neural radiance field;Deep learning;Artificial intelligence;3D Gaussian Splatting;Cage-based Deformation;3D Representation Editing;Animation},
  doi={10.1109/TVCG.2026.3687062}}
```
