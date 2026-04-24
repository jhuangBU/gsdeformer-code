#!/bin/env bash
set -e

bash deforming-nerf-exp-quant-benchmark.sh
bash frosting-exp-quant-benchmark.sh
bash games-exp-quant-benchmark.sh
bash sugar-exp-quant-benchmark.sh