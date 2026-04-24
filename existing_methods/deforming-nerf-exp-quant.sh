#!/usr/bin/env bash
set -e

# Usage: ./deforming-nerf-exp-quant.sh <scene_name>
# Example: ./deforming-nerf-exp-quant.sh quant_mic

if [ $# -lt 1 ]; then
    echo "Usage: $0 <scene_name>"
    exit 1
fi

SCENE=$1
project_root=$(pwd)/..
BASE=$project_root/data/quantitative/$SCENE
DATA_PATH=$BASE/${SCENE}_deformed
CKPT_DIR=$project_root/existing_methods/deforming_nerf/opt/ckpt/${SCENE}
CKPT_PATH=$CKPT_DIR/ckpt.npz
EVAL_PATH=$project_root/existing_methods/deforming_nerf/output/eval_$SCENE

run_python="conda run -n deforming-nerf --no-capture-output python"

CONFIG=${CONFIG:-configs/syn.json}
COORD_TYPE=${COORD_TYPE:-MVC}

batch_size_arg=${BATCH_SIZE:+"--batch_size $BATCH_SIZE"}
coord_batch_size_arg=${COORD_BATCH_SIZE:+"--coord_batch_size $COORD_BATCH_SIZE"}

# Validate paths
[ ! -d "$BASE" ] && { echo "Error: Base directory not found: $BASE"; exit 1; }
[ ! -d "$DATA_PATH" ] && { echo "Error: Deformed dataset not found: $DATA_PATH"; exit 1; }
[ ! -f "$BASE/${SCENE}_proxy.ply" ] && { echo "Error: Source cage not found: $BASE/${SCENE}_proxy.ply"; exit 1; }
[ ! -f "$BASE/${SCENE}_proxy_deformed.ply" ] && { echo "Error: Target cage not found: $BASE/${SCENE}_proxy_deformed.ply"; exit 1; }
[ ! -f "$CKPT_PATH" ] && { echo "Error: Checkpoint not found: $CKPT_PATH"; exit 1; }

echo "Scene: $SCENE"
echo "Checkpoint: $CKPT_PATH"
echo "==========================================="

cd deforming_nerf/opt

# Step 1: Deform and render all train images (filename-aligned with GT)
echo "[1/4] Deforming and rendering train images..."
rm -rf "$EVAL_PATH"
mkdir -p "$EVAL_PATH/test/deforming_nerf/renders"

$run_python render_imgs_deform.py \
  "$CKPT_PATH" "$DATA_PATH" \
  --train \
  -c "$CONFIG" \
  --coord_type "$COORD_TYPE" \
  --auto_near_far \
  --no_vid \
  --cage_source "$BASE/${SCENE}_proxy.ply" \
  --cage_target "$BASE/${SCENE}_proxy_deformed.ply" \
  --cage_scale 0.6667 \
  --output_folder "$EVAL_PATH/test/deforming_nerf/renders" \
  $batch_size_arg \
  $coord_batch_size_arg

# Step 2: Create folder structure for metrics.py
echo "[2/4] Setting up folder structure..."
mkdir -p "$EVAL_PATH/test/deforming_nerf/gt"

# Step 3: Copy ground truth images
echo "[3/4] Copying ground truth images..."
cp "$DATA_PATH"/train/*.png "$EVAL_PATH/test/deforming_nerf/gt/"

# Step 4: Compute metrics (with masks if available)
echo "[4/4] Computing metrics..."
cd "$project_root/gs3d"
MASK_DIR="$BASE/${SCENE}_deformed/masks"
MASK_ARG=""
[ -d "$MASK_DIR" ] && MASK_ARG="--masks $MASK_DIR"
conda run -n cagegaussian --no-capture-output python metrics.py -m "$EVAL_PATH" --background white $MASK_ARG
cd ..

echo "==========================================="
echo "✓ Complete! Results saved to: $EVAL_PATH/results.json"
