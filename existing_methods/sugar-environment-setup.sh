#!/bin/bash
set -euo pipefail

cd sugar
mamba env update -f environment.yml
conda run -n sugar --no-capture-output bash ../sugar-environment-setup-1.sh