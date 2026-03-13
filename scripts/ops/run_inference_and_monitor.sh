#!/usr/bin/env bash
# Run checkpoint inference in the terminal and stream output to stdout + log for monitoring.
# Auto-finds checkpoints under REPO_ROOT if CHECKPOINTS_BASE is not set.
# Usage: ./scripts/run_inference_and_monitor.sh [extra args...]

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
LOG_DIR="${LOG_DIR:-$REPO_ROOT}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/inference_$(date +%Y%m%d_%H%M%S).log}"

VAL_ANN="${VAL_ANNOTATION:-$REPO_ROOT/datasets/cleaned_splits/maxsight_val.json}"
IMAGE_DIR="${IMAGE_DIR:-$REPO_ROOT/datasets}"
OUTPUT="${OUTPUT:-$REPO_ROOT/inference_data.json}"

# Auto-discover checkpoints base: use CHECKPOINTS_BASE if it has at least one checkpoints_*/best_model.pt, else search
need_discover=1
if [ -n "$CHECKPOINTS_BASE" ] && [ -d "$CHECKPOINTS_BASE" ]; then
  n=$(find "$CHECKPOINTS_BASE" -maxdepth 2 -type f -path '*/checkpoints_*/best_model.pt' 2>/dev/null | wc -l)
  [ "$n" -gt 0 ] && need_discover=0
fi
if [ "$need_discover" -eq 1 ]; then
  CHECKPOINTS_BASE=""
  for candidate in "$REPO_ROOT/checkpoints" "$REPO_ROOT/backups" "$REPO_ROOT"; do
    [ -d "$candidate" ] || continue
    first=$(find "$candidate" -maxdepth 2 -type f -path '*/checkpoints_*/best_model.pt' 2>/dev/null | head -1)
    if [ -n "$first" ]; then
      CHECKPOINTS_BASE="$(cd "$(dirname "$(dirname "$first")")" && pwd)"
      break
    fi
  done
  if [ -z "$CHECKPOINTS_BASE" ]; then
    echo "No checkpoints found. Looked in: $REPO_ROOT/checkpoints, $REPO_ROOT/backups, $REPO_ROOT" | tee -a "$LOG_FILE"
    echo "Expected: <base>/checkpoints_<condition>/best_model.pt" | tee -a "$LOG_FILE"
    echo "To create dir layout: python scripts/ops/ensure_checkpoint_layout.py" | tee -a "$LOG_FILE"
    exit 1
  fi
fi
# CHECKPOINTS_BASE is set and has at least one checkpoint

echo "Log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "Val: $VAL_ANN | Image dir: $IMAGE_DIR | Checkpoints: $CHECKPOINTS_BASE" | tee -a "$LOG_FILE"
echo "---" | tee -a "$LOG_FILE"

python scripts/ops/run_checkpoint_inference.py \
  --val-annotation "$VAL_ANN" \
  --image-dir "$IMAGE_DIR" \
  --checkpoints-base "$CHECKPOINTS_BASE" \
  --output "$OUTPUT" \
  "$@" 2>&1 | tee -a "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}

echo "---" | tee -a "$LOG_FILE"
echo "Output JSON: $OUTPUT" | tee -a "$LOG_FILE"
exit "$EXIT_CODE"
