#!/usr/bin/env bash
set -e

# Usage: ./frosting-exp-quant.sh <scene_name>
# Example: ./frosting-exp-quant.sh quant_mic

if [ $# -lt 1 ]; then
    echo "Usage: $0 <scene_name>"
    exit 1
fi

SCENE=$1
project_root=$(pwd)/..
BASE=$project_root/data/quantitative/$SCENE
MODEL_PATH=$project_root/existing_methods/frosting/output/vanilla_gs/${SCENE}
EVAL_PATH=$project_root/existing_methods/frosting/output/eval_$SCENE

run_python_frosting="conda run -n frosting --no-capture-output python"

# Validate paths
[ ! -d "$BASE" ] && { echo "Error: Base directory not found: $BASE"; exit 1; }
[ ! -d "$BASE/$SCENE" ] && { echo "Error: Dataset not found"; exit 1; }
[ ! -d "$BASE/${SCENE}_deformed" ] && { echo "Error: Deformed dataset not found"; exit 1; }
[ ! -d "$MODEL_PATH" ] && { echo "Error: Model directory not found: $MODEL_PATH"; exit 1; }
[ ! -f "$BASE/${SCENE}_proxy.ply" ] && { echo "Error: Source cage not found"; exit 1; }
[ ! -f "$BASE/${SCENE}_proxy_deformed.ply" ] && { echo "Error: Target cage not found"; exit 1; }

MODEL_FINE_PATH=$project_root/existing_methods/frosting/output/refined_frosting/${SCENE}/frostingfine_3Dgs7000_densityestim02_sdfnorm02_level03_decim1000000_depthauto_quantile01_gauss2000000_frostlevel001_proposal30/15000.pt
[ ! -f "$MODEL_FINE_PATH" ] && { echo "Error: Frosting checkpoint not found: $MODEL_FINE_PATH"; exit 1; }

echo "Scene: $SCENE"
echo "==========================================="

# Frosting loads camera poses from `${MODEL_PATH}/cameras.json` (from GS training output).
# For scenes where `${SCENE}_deformed` uses different cameras (e.g. quant_ficus),
# quantitative eval must render with deformed camera poses to match GT.
CAMERAS_JSON="$MODEL_PATH/cameras.json"
CAMERAS_JSON_BAK="$MODEL_PATH/cameras.json.orig"
DEFORMED_TRANSFORMS="$BASE/${SCENE}_deformed/transforms_train.json"
CONVERT_SCRIPT="$project_root/existing_methods/frosting/convert_transforms_to_cameras_json.py"

restore_cameras_json() {
  if [ -f "$CAMERAS_JSON_BAK" ]; then
    mv -f "$CAMERAS_JSON_BAK" "$CAMERAS_JSON"
  fi
}
trap restore_cameras_json EXIT

echo "[0/4] Patching cameras.json to use deformed camera poses..."
[ ! -f "$CAMERAS_JSON" ] && { echo "Error: Missing model cameras.json: $CAMERAS_JSON"; exit 1; }
[ ! -f "$DEFORMED_TRANSFORMS" ] && { echo "Error: Missing deformed transforms: $DEFORMED_TRANSFORMS"; exit 1; }
[ ! -f "$CONVERT_SCRIPT" ] && { echo "Error: Missing converter script: $CONVERT_SCRIPT"; exit 1; }
cp -f "$CAMERAS_JSON" "$CAMERAS_JSON_BAK"
$run_python_frosting "$CONVERT_SCRIPT" \
  --source_path "$BASE/${SCENE}_deformed" \
  --model_path "$MODEL_PATH" \
  --white_background

cd frosting

# Step 1: Deform and render all test images
echo "[1/4] Deforming and rendering test images..."
rm -rf $EVAL_PATH
mkdir -p $EVAL_PATH/test/frosting/renders

$run_python_frosting visualize_results.py \
  --src_cage "$BASE/${SCENE}_proxy.ply" \
  --dst_cage "$BASE/${SCENE}_proxy_deformed.ply" \
  --source_path "$BASE/${SCENE}_deformed" \
  --gs_folder "$MODEL_PATH" \
  --frosting_checkpoint "$MODEL_FINE_PATH" \
  --cam_idx -1 \
  --output_folder "$EVAL_PATH/test/frosting/renders" \
  --white_background \
  --deformation_threshold 0.0

# Step 2: Create folder structure for metrics.py
echo "[2/4] Setting up folder structure..."
mkdir -p $EVAL_PATH/test/frosting/gt

# Step 3: Copy ground truth images
echo "[3/4] Copying ground truth images..."
cp $BASE/${SCENE}_deformed/train/*.png $EVAL_PATH/test/frosting/gt/

# Step 4: Compute metrics (with masks if available)
echo "[4/4] Computing metrics..."
cd $project_root/gs3d
MASK_DIR="$BASE/${SCENE}_deformed/masks"
MASK_ARG=""
[ -d "$MASK_DIR" ] && MASK_ARG="--masks $MASK_DIR"
conda run -n cagegaussian --no-capture-output python metrics.py -m "$EVAL_PATH" --background white $MASK_ARG
cd ..

echo "==========================================="
echo "✓ Complete! Results saved to: $EVAL_PATH/results.json"
