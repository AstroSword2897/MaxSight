#!/usr/bin/env bash
# MLX-style / T5-style full convergence run for Mac
# Uses: DEVICE=mlx (→ CPU), batch 4 + grad accum 4 → effective batch 16,
#       separate backbone/head LRs, cosine scheduler + warmup, checkpoint every 5 epochs, early stopping.
# Run: ./scripts/run_mlx_style_training.sh [--no-export]
# Background: nohup ./scripts/run_mlx_style_training.sh --no-export > training_run.log 2>&1 &
# Monitor: tail -f training_run.log
# Resume: RESUME_FROM=checkpoints/last_checkpoint.pt ./scripts/run_mlx_style_training.sh --no-export

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

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
export LR="${LR:-1e-4}"

exec ./scripts/run_production_training.sh "$@"
