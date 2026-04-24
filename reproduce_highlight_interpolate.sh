#!/usr/bin/env bash
#
# Replicate the `infer_gsdeformer_exp_highlight_interpolate` experiment
# from a vanilla Ubuntu 20.04.1 LTS host with an NVIDIA GPU.
#
# Prerequisites (must exist on the host before running this script):
#   - NVIDIA driver >= 510 (CUDA 11.6 compatible). Verify with `nvidia-smi`.
#   - `sudo` access for `apt-get install`.
#
# Usage (run from the repo root):
#   bash reproduce_highlight_interpolate.sh
#
# Outputs:
#   exp-qual-highlight-interpolate/   per-frame rendered PNGs
#   stacked.png                        compiled 2x5 grid figure

set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

BUNDLE_URL="https://huggingface.co/datasets/jjhuangbu/gsdeformer-data/resolve/main/nerf-lego-reproduction.zip"

############################################
# 1. System packages
############################################
echo "=== [1/5] apt packages ==="
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates curl wget git git-lfs unzip bzip2 xvfb \
    libgl1 libegl1 libgomp1 libxml2 \
    libx11-6 libxext6 libxrender1 libxrandr2 libxcursor1 libxinerama1 libxi6 libxxf86vm1
sudo git lfs install --system

############################################
# 2. Miniforge + just
############################################
echo "=== [2/5] miniforge + just ==="
if [[ ! -d "$HOME/miniforge3" ]]; then
    wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
        -O /tmp/miniforge.sh
    bash /tmp/miniforge.sh -b -p "$HOME/miniforge3"
    rm /tmp/miniforge.sh
fi
# shellcheck disable=SC1091
source "$HOME/miniforge3/etc/profile.d/conda.sh"

if ! command -v just >/dev/null 2>&1; then
    mkdir -p "$HOME/.local/bin"
    curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh \
        | bash -s -- --to "$HOME/.local/bin"
fi
export PATH="$HOME/.local/bin:$PATH"

############################################
# 3. Conda env (CUDA 11.6, PyTorch 1.12.1, Taichi, Open3D, gcc-11,
#    and the three local CUDA rasterizer extensions)
############################################
echo "=== [3/5] conda env ==="
just environment_setup
conda activate cagegaussian

############################################
# 4. Datasets + pretrained 3DGS lego bundle
############################################
echo "=== [4/5] datasets ==="

# gsdeformer-data on HuggingFace provides cages, cameras, and baselines.
if [[ ! -d data/cameras_qualitative ]]; then
    rm -rf data_hf
    git clone https://huggingface.co/datasets/jjhuangbu/gsdeformer-data data_hf
    mkdir -p data
    (cd data_hf && cp -r . ../data/)
    rm -rf data_hf
fi

# Preprocessed lego dataset + trained 3DGS model, packed as a single zip on HF.
# Layout inside the zip:
#   nerf-lego-reproduction/nerf_lego/          -> data/deforming_nerf-data/nerf_lego/
#   nerf-lego-reproduction/gs3d-output/nerf_lego/ -> gs3d/output/nerf_lego/
if [[ ! -d gs3d/output/nerf_lego/point_cloud || ! -d data/deforming_nerf-data/nerf_lego ]]; then
    wget -q "$BUNDLE_URL" -O /tmp/nerf-lego-reproduction.zip
    STAGE="$(mktemp -d)"
    unzip -q /tmp/nerf-lego-reproduction.zip -d "$STAGE"

    mkdir -p data/deforming_nerf-data gs3d/output
    rm -rf data/deforming_nerf-data/nerf_lego gs3d/output/nerf_lego
    mv "$STAGE/nerf-lego-reproduction/nerf_lego"        data/deforming_nerf-data/nerf_lego
    mv "$STAGE/nerf-lego-reproduction/gs3d-output/nerf_lego" gs3d/output/nerf_lego

    rm -rf "$STAGE" /tmp/nerf-lego-reproduction.zip
fi

############################################
# 5. Run inference (headless via xvfb)
############################################
echo "=== [5/5] inference ==="
xvfb-run -a just infer_gsdeformer_exp_highlight_interpolate

echo
echo "=== done ==="
echo "Per-frame outputs: $REPO_ROOT/exp-qual-highlight-interpolate/"
echo "Compiled figure:   $REPO_ROOT/stacked.png"
