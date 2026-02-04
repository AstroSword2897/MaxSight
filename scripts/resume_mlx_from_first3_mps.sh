#!/usr/bin/env bash
# Resume from first 3 MPS epochs with full MLX-style config (batch 4, grad acc 4, LR backbone/head, cosine, etc.).
# Use this after you stopped an MPS run after ~3 epochs and want to continue with the new recipe.
#
# Step 1: Back up your current checkpoints (from the MPS run) into backups/first3_epochs
# Step 2: Run this script — it uses that backup and resumes with MLX-style settings (model-only load so new optimizer/scheduler apply).
#
# Usage:
#   ./scripts/resume_mlx_from_first3_mps.sh [--no-export]
#
# Prereqs:
#   - You have checkpoints from the first 3 MPS epochs either in checkpoints/ or already in backups/first3_epochs/
#   - If still in checkpoints/, the script will copy them to backups/first3_epochs for you.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

BACKUP_DIR="${BACKUP_DIR:-backups/first3_epochs}"
CKPT_DIR="${CHECKPOINT_DIR:-./checkpoints}"

mkdir -p "$BACKUP_DIR"

# If we don't have last_checkpoint.pt in backup yet, copy from checkpoints/
if [[ ! -f "$BACKUP_DIR/last_checkpoint.pt" ]] && [[ -f "$CKPT_DIR/last_checkpoint.pt" ]]; then
  echo "Copying first-3-epoch checkpoints from $CKPT_DIR to $BACKUP_DIR ..."
  cp "$CKPT_DIR/last_checkpoint.pt" "$BACKUP_DIR/"
  [[ -f "$CKPT_DIR/best_model.pt" ]] && cp "$CKPT_DIR/best_model.pt" "$BACKUP_DIR/"
  echo "  Done. Resuming from $BACKUP_DIR/last_checkpoint.pt with MLX-style config."
elif [[ ! -f "$BACKUP_DIR/last_checkpoint.pt" ]]; then
  echo "ERROR: No checkpoint found at $BACKUP_DIR/last_checkpoint.pt or $CKPT_DIR/last_checkpoint.pt"
  echo "  Run 3 epochs on MPS first, then run this script (or copy last_checkpoint.pt to $BACKUP_DIR/)."
  exit 1
fi

# Optional: save loss lines from current log
if [[ -f training_run.log ]]; then
  grep -E "Epoch|Loss|Train Loss|Val Loss" training_run.log > "$BACKUP_DIR/epoch_losses.txt" 2>/dev/null || true
fi

export DEVICE="${DEVICE:-mlx}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export GRAD_ACC="${GRAD_ACC:-4}"
export EPOCHS="${EPOCHS:-50}"
export NUM_WORKERS="${NUM_WORKERS:-2}"
export LR_BACKBONE="${LR_BACKBONE:-1e-5}"
export LR_HEAD="${LR_HEAD:-1e-4}"
export SCHEDULER="${SCHEDULER:-cosine}"
export WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
export EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-10}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-5}"
export RESUME_FROM="$BACKUP_DIR/last_checkpoint.pt"
export RESUME_MODEL_ONLY=1

echo "============================================================"
echo "Resume from first 3 MPS epochs → MLX-style (model only load)"
echo "============================================================"
echo "  RESUME_FROM=$RESUME_FROM"
echo "  RESUME_MODEL_ONLY=1 (new LRs, scheduler, batch 4, grad acc 4)"
echo "  DEVICE=$DEVICE  BATCH_SIZE=$BATCH_SIZE  GRAD_ACC=$GRAD_ACC  EPOCHS=$EPOCHS"
echo "============================================================"

exec ./scripts/run_production_training.sh "$@"
