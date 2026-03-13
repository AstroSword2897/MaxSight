#!/usr/bin/env bash
# MaxSight Full Production Training — One Shot
# Runs: env check → dataset check → optional data prep → training → optional export.
# Usage: ./scripts/ops/run_production_training.sh [--skip-env] [--skip-data-check] [--no-export] [--dry-run]

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Defaults (override with env or edit)
# Optional: run AutoML first: python scripts/AutoMLType.py --data-dir ... --train-annotation ... --val-annotation ... --image-dir ...
# Then set HYPERPARAMETERS=checkpoints_tuning/best_hyperparameters.json for tuned LR/weight_decay/loss weights.
DATA_DIR="${DATA_DIR:-datasets/coco_raw}"
SPLITS_DIR="${SPLITS_DIR:-datasets/cleaned_splits}"
TRAIN_ANN="${TRAIN_ANN:-$SPLITS_DIR/maxsight_train.json}"
VAL_ANN="${VAL_ANN:-$SPLITS_DIR/maxsight_val.json}"
IMAGE_DIR="${IMAGE_DIR:-$DATA_DIR}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints}"
HYPERPARAMETERS="${HYPERPARAMETERS:-}"  # optional: best_hyperparameters.json from AutoMLType.py
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
# MLX-style / T5-style: gradient accumulation, separate LRs, scheduler, early stop
GRAD_ACC="${GRAD_ACC:-1}"
LR_BACKBONE="${LR_BACKBONE:-}"
LR_HEAD="${LR_HEAD:-}"
SCHEDULER="${SCHEDULER:-cosine}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-10}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-0}"
# auto = use CUDA on Linux if available, else CPU (MPS disabled due to errors); mlx = CPU
DEVICE="${DEVICE:-auto}"
DO_EXPORT="${DO_EXPORT:-1}"
# Resume: set RESUME=1 to use latest in CHECKPOINT_DIR, or RESUME_FROM=/path/to/last_checkpoint.pt to continue on another GPU
RESUME="${RESUME:-0}"
RESUME_FROM="${RESUME_FROM:-}"
RESUME_MODEL_ONLY="${RESUME_MODEL_ONLY:-0}"  # 1 = load only model+epoch from checkpoint; use current LRs/scheduler
SKIP_ENV=0
SKIP_DATA_CHECK=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-env)       SKIP_ENV=1; shift ;;
    --skip-data-check) SKIP_DATA_CHECK=1; shift ;;
    --no-export)      DO_EXPORT=0; shift ;;
    --dry-run)        DRY_RUN=1; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

echo "============================================================"
echo "MaxSight Production Training — One Shot"
echo "============================================================"
echo "  DATA_DIR=$DATA_DIR"
echo "  TRAIN_ANN=$TRAIN_ANN"
echo "  VAL_ANN=$VAL_ANN"
echo "  IMAGE_DIR=$IMAGE_DIR"
echo "  EPOCHS=$EPOCHS  BATCH_SIZE=$BATCH_SIZE  NUM_WORKERS=$NUM_WORKERS  GRAD_ACC=$GRAD_ACC  LR=$LR"
echo "  SCHEDULER=$SCHEDULER  WARMUP_EPOCHS=$WARMUP_EPOCHS  EARLY_STOP=$EARLY_STOPPING_PATIENCE  CHECKPOINT_INTERVAL=$CHECKPOINT_INTERVAL"
[[ -n "$LR_BACKBONE" ]] && echo "  LR_BACKBONE=$LR_BACKBONE  LR_HEAD=$LR_HEAD"
echo "  DEVICE=$DEVICE  DO_EXPORT=$DO_EXPORT"
[[ -n "$RESUME_FROM" ]] && echo "  RESUME_FROM=$RESUME_FROM"
[[ "$RESUME" -eq 1 ]] && echo "  RESUME=1 (latest checkpoint in $CHECKPOINT_DIR)"
echo "============================================================"

# --- 1) Environment check ---
if [[ "$SKIP_ENV" -eq 0 ]]; then
  echo "[1/5] Environment check..."
  pip install --upgrade pip -q
  pip install -r requirements.txt -q
  python -c '
import torch
if torch.cuda.is_available():
    print("  CUDA:", torch.cuda.get_device_name(0))
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    print("  MPS (Apple GPU): available")
else:
    print("  PyTorch OK (device=cpu)")
'
  echo "  Environment OK."
else
  echo "[1/5] Environment check skipped."
fi

# --- 2) Dataset check (and optional gather) ---
if [[ "$SKIP_DATA_CHECK" -eq 0 ]]; then
  echo "[2/5] Dataset check..."
  if [[ ! -d "$IMAGE_DIR" ]]; then
    echo "  Image dir missing: $IMAGE_DIR"
    echo "  Run: python scripts/ops/gather_training_data.py [--data-dir $DATA_DIR]"
    exit 1
  fi
  if [[ ! -f "$TRAIN_ANN" ]] || [[ ! -f "$VAL_ANN" ]]; then
    echo "  Splits missing. Running gather_training_data.py..."
    python scripts/ops/gather_training_data.py --data-dir "$DATA_DIR" --splits-dir "$SPLITS_DIR"
  fi
  echo "  Data OK: $TRAIN_ANN, $VAL_ANN"
else
  echo "[2/5] Dataset check skipped."
fi

# --- 3) Optional: Phase 3 data pipeline validation ---
if [[ "$DRY_RUN" -eq 0 ]] && [[ -f "scripts/ops/validate_data_pipeline.py" ]]; then
  echo "[3/5] Data pipeline validation (optional)..."
  if python scripts/ops/validate_data_pipeline.py --train-annotation "$TRAIN_ANN" --image-dir "$IMAGE_DIR" 2>/dev/null; then
    echo "  Data pipeline validation OK."
  else
    echo "  Data pipeline validation skipped or failed (non-fatal)."
  fi
else
  echo "[3/5] Data pipeline validation skipped."
fi

# --- 4) Training ---
echo "[4/5] Training..."
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "  [DRY RUN] Would run:"
    echo "  python scripts/ops/train_maxsight.py \\"
  echo "    --data-dir $DATA_DIR \\"
  echo "    --train-annotation $TRAIN_ANN \\"
  echo "    --val-annotation $VAL_ANN \\"
  echo "    --image-dir $IMAGE_DIR \\"
  echo "    --checkpoint-dir $CHECKPOINT_DIR \\"
  echo "    --epochs $EPOCHS --batch-size $BATCH_SIZE --num-workers $NUM_WORKERS \\"
  echo "    --learning-rate $LR --weight-decay $WEIGHT_DECAY \\"
  echo "    --device $DEVICE --fp16 --use-gradnorm \\"
  [[ -n "$HYPERPARAMETERS" ]] && [[ -f "$HYPERPARAMETERS" ]] && echo "    --hyperparameters $HYPERPARAMETERS \\"
  echo ""
  exit 0
fi

HP_ARGS=()
[[ -n "$HYPERPARAMETERS" ]] && [[ -f "$HYPERPARAMETERS" ]] && HP_ARGS=(--hyperparameters "$HYPERPARAMETERS")
RESUME_ARGS=()
[[ "$RESUME" -eq 1 ]] && RESUME_ARGS=(--resume)
[[ -n "$RESUME_FROM" ]] && [[ -f "$RESUME_FROM" ]] && RESUME_ARGS=(--resume-from "$RESUME_FROM")
[[ "$RESUME_MODEL_ONLY" -eq 1 ]] && RESUME_ARGS+=("--resume-model-only")
LR_EXTRA=()
[[ -n "$LR_BACKBONE" ]] && LR_EXTRA+=(--lr-backbone "$LR_BACKBONE")
[[ -n "$LR_HEAD" ]] && LR_EXTRA+=(--lr-head "$LR_HEAD")

python scripts/ops/train_maxsight.py \
  --data-dir "$DATA_DIR" \
  --train-annotation "$TRAIN_ANN" \
  --val-annotation "$VAL_ANN" \
  --image-dir "$IMAGE_DIR" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --learning-rate "$LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --grad-accumulation-steps "$GRAD_ACC" \
  --scheduler-type "$SCHEDULER" \
  --warmup-epochs "$WARMUP_EPOCHS" \
  --early-stopping-patience "$EARLY_STOPPING_PATIENCE" \
  --checkpoint-interval "$CHECKPOINT_INTERVAL" \
  --device "$DEVICE" \
  --fp16 \
  --use-gradnorm \
  "${LR_EXTRA[@]}" \
  "${RESUME_ARGS[@]}" \
  "${HP_ARGS[@]}"

echo "  Training step finished."

# --- 5) Optional export ---
if [[ "$DO_EXPORT" -eq 1 ]]; then
  echo "[5/5] Export (JIT)..."
  BEST="$CHECKPOINT_DIR/best.pt"
  if [[ -f "$BEST" ]]; then
    python -m ml.training.export --checkpoint "$BEST" --format jit --output exports/maxsight_jit.pt --device "$DEVICE" || true
  else
    echo "  No $BEST found; skipping export. Use: python -m ml.training.export --checkpoint <path> --format jit|onnx|coreml --output <path>"
  fi
else
  echo "[5/5] Export skipped (--no-export)."
fi

echo "============================================================"
echo "Done. Checkpoints: $CHECKPOINT_DIR"
echo "Evaluation: no scripts/test_maxsight.py; use pytest tests/ or scripts/archive/ for benchmarks."
echo "============================================================"
