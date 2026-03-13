#!/usr/bin/env bash
# Improve mAP for all condition models: find trained checkpoints, sweep confidence/NMS, then run inference with best params.
# Uses trained weights only (skips if only minimal/placeholder checkpoints are found).
set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# 1) Resolve checkpoint base: env, or discover via find_trained_checkpoints.py
if [ -n "$CHECKPOINTS_BASE" ] && [ -d "$CHECKPOINTS_BASE" ]; then
  BASE="$CHECKPOINTS_BASE"
else
  BASE=$(python scripts/ops/find_trained_checkpoints.py 2>/dev/null) || true
  if [ -z "$BASE" ] || [ ! -d "$BASE" ]; then
    echo "Set CHECKPOINTS_BASE to the folder containing checkpoints_<condition>/best_model.pt (trained weights)."
    echo "Example: CHECKPOINTS_BASE=\"\$HOME/Google Drive/My Drive/MaxSight\" ./scripts/improve_map_all_models.sh"
    echo "Colab: /content/drive/MyDrive/MaxSight"
    exit 1
  fi
fi

VAL_ANN="${VAL_ANNOTATION:-$REPO_ROOT/datasets/cleaned_splits/maxsight_val.json}"
IMAGE_DIR="${IMAGE_DIR:-$REPO_ROOT/datasets}"
OUTPUT="${OUTPUT:-$REPO_ROOT/inference_data.json}"

echo "Checkpoints base: $BASE"
echo "Val: $VAL_ANN | Image dir: $IMAGE_DIR"
echo ""

# 2) Sweep confidence and NMS IoU to find best mAP (no retraining)
echo "=== Sweeping confidence and NMS IoU to improve mAP ==="
SWEEP_OUTPUT=$(mktemp)
python scripts/research_archive/optimize_inference.py \
  --val-annotation "$VAL_ANN" \
  --image-dir "$IMAGE_DIR" \
  --checkpoints-base "$BASE" \
  ${CONDITIONS:+--conditions $CONDITIONS} \
  ${MAX_BATCHES:+--max-batches $MAX_BATCHES} 2>&1 | tee "$SWEEP_OUTPUT"

# Parse best confidence and nms_iou from optimize_inference output
BEST_CONF="0.05"
BEST_NMS="0.5"
while IFS= read -r line; do
  if [[ "$line" =~ confidence:\ ([0-9.]+) ]]; then BEST_CONF="${BASH_REMATCH[1]}"; fi
  if [[ "$line" =~ nms_iou:\ ([0-9.]+) ]]; then BEST_NMS="${BASH_REMATCH[1]}"; fi
done < "$SWEEP_OUTPUT"
rm -f "$SWEEP_OUTPUT"

# 3) Run full inference with best params for all conditions
echo ""
echo "=== Full inference with best params (confidence=$BEST_CONF, nms_iou=$BEST_NMS) ==="
python scripts/ops/run_checkpoint_inference.py \
  --val-annotation "$VAL_ANN" \
  --image-dir "$IMAGE_DIR" \
  --checkpoints-base "$BASE" \
  --confidence "$BEST_CONF" \
  --nms-iou "$BEST_NMS" \
  --output "$OUTPUT"
echo "Wrote $OUTPUT"
