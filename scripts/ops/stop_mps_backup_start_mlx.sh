#!/usr/bin/env bash
# Step 1: Stop current MPS training (SIGINT). Step 2: Backup checkpoints & log. Step 3: Start MLX fine-tuning with resume. Run from repo root.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

BACKUP_DIR="${BACKUP_DIR:-backups/first3_epochs}"
CKPT_DIR="${CHECKPOINT_DIR:-./checkpoints}"

echo "============================================================"
echo "Step 1: Stop current MPS run (SIGINT)"
echo "============================================================"
PID=""
for p in $(pgrep -f "run_production_training.sh" 2>/dev/null); do
  # Prefer the shell script, not the python child
  if ps -p "$p" -o comm= 2>/dev/null | grep -q run_production; then
    PID=$p
    break
  fi
done
if [[ -z "$PID" ]]; then
  for p in $(pgrep -f "train_maxsight.py" 2>/dev/null); do
    PID=$p
    break
  done
fi
if [[ -n "$PID" ]]; then
  echo "Sending SIGINT to PID $PID ..."
  kill -SIGINT "$PID" 2>/dev/null || true
  echo "Waiting 15s for checkpoint to save ..."
  sleep 15
else
  echo "No run_production_training.sh or train_maxsight.py process found. Proceeding with backup/start."
fi

echo "============================================================"
echo "Step 2: Backup checkpoints and log"
echo "============================================================"
mkdir -p "$BACKUP_DIR"
if [[ -f "$CKPT_DIR/last_checkpoint.pt" ]]; then
  cp "$CKPT_DIR/last_checkpoint.pt" "$BACKUP_DIR/"
  echo "  Copied last_checkpoint.pt"
fi
if [[ -f "$CKPT_DIR/best_model.pt" ]]; then
  cp "$CKPT_DIR/best_model.pt" "$BACKUP_DIR/"
  echo "  Copied best_model.pt"
fi
if [[ -f training_run.log ]]; then
  if grep -q "Epoch 4" training_run.log 2>/dev/null; then
    head -n "$(grep -n "Epoch 4" training_run.log | head -1 | cut -d: -f1)" training_run.log > "$BACKUP_DIR/epoch_losses.txt"
  else
    grep -E "Epoch|Loss|Train Loss|Val Loss" training_run.log > "$BACKUP_DIR/epoch_losses.txt" 2>/dev/null || cp training_run.log "$BACKUP_DIR/epoch_losses.txt"
  fi
  echo "  Saved epoch_losses.txt"
fi

echo "============================================================"
echo "Step 3: Start MLX fine-tuning (resume from backup)"
echo "============================================================"
if [[ ! -f "$BACKUP_DIR/last_checkpoint.pt" ]]; then
  echo "ERROR: No $BACKUP_DIR/last_checkpoint.pt. Run at least one epoch first."
  exit 1
fi

export DEVICE=mlx
export BATCH_SIZE=4
export GRAD_ACC=4
export EPOCHS=50
export NUM_WORKERS=2
export LR_BACKBONE=1e-5
export LR_HEAD=1e-4
export SCHEDULER=cosine
export WARMUP_EPOCHS=5
export CHECKPOINT_INTERVAL=5
export RESUME_FROM="$BACKUP_DIR/last_checkpoint.pt"
export RESUME_MODEL_ONLY=1

exec ./scripts/ops/run_production_training.sh --no-export "$@"
