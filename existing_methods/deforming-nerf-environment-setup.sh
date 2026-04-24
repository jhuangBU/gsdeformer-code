#!/bin/bash
set -euo pipefail

cd deforming_nerf
mamba env update -f environment.yml
conda run -n deforming-nerf --no-capture-output pip install .