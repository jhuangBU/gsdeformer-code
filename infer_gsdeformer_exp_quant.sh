#!/bin/bash

set -e

export LIBGL_ALWAYS_SOFTWARE=1

# Usage: ./infer_gsdeformer_exp_quant_v2.sh <scene_name>
# Example: ./infer_gsdeformer_exp_quant_v2.sh mic

if [ $# -lt 1 ]; then
    echo "Usage: $0 <scene_name>"
    exit 1
fi

SCENE=$1
BASE=data/quantitative/$SCENE
MODEL_PATH=gs3d/output/$SCENE
EVAL_PATH=gs3d/output/eval_$SCENE

# Validate paths
[ ! -d "$BASE" ] && { echo "Error: Base directory not found: $BASE"; exit 1; }
[ ! -d "$BASE/$SCENE" ] && { echo "Error: Dataset not found"; exit 1; }
[ ! -d "$BASE/${SCENE}_deformed" ] && { echo "Error: Deformed dataset not found"; exit 1; }
[ ! -d "$MODEL_PATH" ] && { echo "Error: Model directory not found"; exit 1; }
[ ! -f "$BASE/${SCENE}_proxy.ply" ] && { echo "Error: Source cage not found"; exit 1; }
[ ! -f "$BASE/${SCENE}_proxy_deformed.ply" ] && { echo "Error: Target cage not found"; exit 1; }

# Find latest iteration
LATEST_ITER=$(ls -v $MODEL_PATH/point_cloud/ | grep "^iteration_" | tail -1 | sed 's/iteration_//')
[ -z "$LATEST_ITER" ] && { echo "Error: No iteration found"; exit 1; }

echo "Scene: $SCENE"
echo "Latest iteration: $LATEST_ITER"
echo "==========================================="

# Step 1: Deform and render all test images using cli.py
echo "[1/4] Deforming and rendering test images..."
rm -rf $EVAL_PATH
mkdir -p $EVAL_PATH/renders_raw

python -m gsdeformer.editorv2.cli \
    source_path=$BASE/${SCENE}_deformed \
    model_path=$MODEL_PATH \
    cage_path=$BASE/${SCENE}_proxy.ply \
    dst_cage_path=$BASE/${SCENE}_proxy_deformed.ply \
    cam_idx=-1 \
    algorithm=all \
    expname=quant_${SCENE} \
    output_path=$EVAL_PATH/renders_raw \
    save_model=false \
    iteration=$LATEST_ITER \
    use_camera_resolution=true \
    white_bg=true

# Step 2: Create folder structure for metrics.py
echo "[2/4] Setting up folder structure..."
mkdir -p $EVAL_PATH/test/ours_30000/renders
mkdir -p $EVAL_PATH/test/ours_30000/gt

# Step 3: Copy rendered images (strip _deformed suffix)
echo "[3/4] Organizing rendered images..."
for f in $EVAL_PATH/renders_raw/*_deformed.png; do
    if [ -f "$f" ]; then
        filename=$(basename "$f")
        # Strip _deformed.png and add .png back
        newname="${filename%_deformed.png}.png"
        cp "$f" "$EVAL_PATH/test/ours_30000/renders/$newname"
    fi
done

# Copy ground truth images
echo "Copying ground truth images..."
cp $BASE/${SCENE}_deformed/train/*.png $EVAL_PATH/test/ours_30000/gt/

# Step 4: Compute metrics (with masks if available)
echo "[4/4] Computing metrics..."
cd gs3d
MASK_DIR="../$BASE/${SCENE}_deformed/masks"
MASK_ARG=""
[ -d "$MASK_DIR" ] && MASK_ARG="--masks $MASK_DIR"
python metrics.py -m output/eval_$SCENE --background white $MASK_ARG
cd ..

# Export to CSV
python -c "
import json, csv
with open('$EVAL_PATH/results.json') as f:
    data = json.load(f)
method = list(data.keys())[0]
m = data[method]
with open('$EVAL_PATH/results.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['scene', 'SSIM', 'PSNR', 'LPIPS'])
    w.writerow(['$SCENE', f\"{m['SSIM']:.6f}\", f\"{m['PSNR']:.6f}\", f\"{m['LPIPS']:.6f}\"])
print(f\"Results: SSIM={m['SSIM']:.6f}, PSNR={m['PSNR']:.6f}, LPIPS={m['LPIPS']:.6f}\")
"

echo "==========================================="
echo "✓ Complete! Results saved to: $EVAL_PATH/results.csv"
