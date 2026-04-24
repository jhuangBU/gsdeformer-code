#!/bin/bash
set -euo pipefail

cd vrgs
conda env update -f environment.yaml

mkdir build
cd build

conda run -n vrgs --no-capture-output cmake ..
conda run -n vrgs --no-capture-output cmake --build . -j 8