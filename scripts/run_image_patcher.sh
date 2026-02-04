#!/bin/bash
# Run the image patcher in the background while training continues
#
# Usage:
#   ./scripts/run_image_patcher.sh          # Patch all splits
#   ./scripts/run_image_patcher.sh train    # Patch train only
#   ./scripts/run_image_patcher.sh val      # Patch val only

set -e

SPLIT=${1:-all}
LOG_FILE="image_patcher_${SPLIT}.log"

echo "Starting COCO image patcher for split: ${SPLIT}"
echo "Logging to: ${LOG_FILE}"
echo "Training can continue running - this runs independently"
echo ""

# Run patcher with output to log file
python3 scripts/patch_missing_images.py \
    --split "${SPLIT}" \
    --workers 4 \
    --root /Users/nani/2026-Prototype \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "✓ Patching complete! Check ${LOG_FILE} for details"
