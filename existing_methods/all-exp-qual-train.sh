#!/bin/env bash
set -e

bash sugar-train.sh
bash games-train.sh
bash frosting-train.sh
bash deforming-nerf-train.sh