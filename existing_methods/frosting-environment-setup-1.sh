#!/bin/bash
set -euo pipefail

export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/targets/x86_64-linux/include

# Install 3D Gaussian Splatting rasterizer
echo "[INFO] Installing diff-gaussian-rasterization"
cd gaussian_splatting/submodules/diff-gaussian-rasterization/
pip install -e .

# Install simple-knn
echo "[INFO] Installing simple-knn..."
cd ../simple-knn/
pip install -e .

cd ../../../

# Install Nvdiffrast
echo "[INFO] Installing Nvdiffrast..."
if [ ! -d "nvdiffrast" ]; then
  git clone https://github.com/NVlabs/nvdiffrast
fi
cd nvdiffrast
pip install .