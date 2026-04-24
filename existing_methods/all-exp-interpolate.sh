#!/bin/env bash
set -e

bash sugar-exp-interpolate.sh
bash games-exp-interpolate.sh
bash deforming-nerf-exp-interpolate.sh
bash frosting-exp-interpolate.sh