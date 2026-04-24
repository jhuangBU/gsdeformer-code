#!/bin/bash

set -euo pipefail

cd frosting
conda env update -f environment.yml
conda run -n frosting --no-capture-output bash ../frosting-environment-setup-1.sh