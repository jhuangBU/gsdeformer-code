#!/bin/bash
set -euo pipefail

export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/targets/x86_64-linux/include
pip install -e gaussian_splatting/submodules/diff-gaussian-rasterization
pip install -e gaussian_splatting/submodules/simple-knn