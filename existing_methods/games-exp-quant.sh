#!/usr/bin/env bash
set -e

# Usage: ./games-exp-quant.sh <scene_name>
# Example: ./games-exp-quant.sh quant_mic

if [ $# -lt 1 ]; then
    echo "Usage: $0 <scene_name>"
    exit 1
fi

SCENE=$1
project_root=$(pwd)/..
BASE=$project_root/data/quantitative/$SCENE
MODEL_PATH=$project_root/existing_methods/gaussian_mesh_splatting/output/${SCENE}
RENDER_OUTPUT=$MODEL_PATH/train/ours_30000/cage_deformed_gs_points
GT_OUTPUT=$MODEL_PATH/train/ours_30000/gt
EVAL_PATH=$project_root/existing_methods/gaussian_mesh_splatting/output/eval_$SCENE

run_python_games="conda run -n gaussian_splatting_mesh --no-capture-output python"

# Validate paths
[ ! -d "$BASE" ] && { echo "Error: Base directory not found: $BASE"; exit 1; }
[ ! -d "$BASE/$SCENE" ] && { echo "Error: Dataset not found"; exit 1; }
[ ! -d "$BASE/${SCENE}_deformed" ] && { echo "Error: Deformed dataset not found"; exit 1; }
[ ! -d "$MODEL_PATH" ] && { echo "Error: Model directory not found: $MODEL_PATH"; exit 1; }
[ ! -f "$BASE/${SCENE}_proxy.ply" ] && { echo "Error: Source cage not found"; exit 1; }
[ ! -f "$BASE/${SCENE}_proxy_deformed.ply" ] && { echo "Error: Target cage not found"; exit 1; }

echo "Scene: $SCENE"
echo "==========================================="

cd gaussian_mesh_splatting

# Step 1: Render all test images using default script behavior
echo "[1/3] Rendering all test images..."
rm -rf $MODEL_PATH/test

$run_python_games -m scripts.render_deform_by_cage \
  -s "$BASE/${SCENE}_deformed" \
  -m "$MODEL_PATH" \
  --iteration 30000 \
  --skip_test \
  --src_cage "$BASE/${SCENE}_proxy.ply" \
  --dst_cage "$BASE/${SCENE}_proxy_deformed.ply"

# Step 2: Reorganize for metrics.py
echo "[2/3] Organizing files for evaluation..."
mkdir -p $EVAL_PATH/test/games

# Copy renders and GT (numeric filenames from script)
mv $RENDER_OUTPUT $EVAL_PATH/test/games/renders
mv $GT_OUTPUT $EVAL_PATH/test/games/gt

# Step 3: Compute metrics (with masks if available)
echo "[3/3] Computing metrics..."
cd $project_root/gs3d
MASK_DIR="$BASE/${SCENE}_deformed/masks"
MASK_ARG=""
[ -d "$MASK_DIR" ] && MASK_ARG="--masks $MASK_DIR"
conda run -n cagegaussian --no-capture-output python metrics.py -m "$EVAL_PATH" --background white $MASK_ARG
cd ../existing_methods

echo "==========================================="
echo "✓ Complete! Results saved to: $EVAL_PATH/results.json"
