#!/bin/bash
# Optimized training for M3 Pro (11 cores, 18GB RAM)

cd /Users/nani/2026-Prototype

python scripts/train_maxsight.py \
  --data-dir datasets/coco_raw \
  --train-annotation datasets/cleaned_splits/maxsight_train.json \
  --val-annotation datasets/cleaned_splits/maxsight_val.json \
  --image-dir datasets/coco_raw \
  --checkpoint-dir ./checkpoints \
  --epochs 50 \
  --batch-size 8 \
  --num-workers 6 \
  --learning-rate 1e-4 \
  --weight-decay 1e-4 \
  --grad-accumulation-steps 2 \
  --scheduler-type cosine \
  --warmup-epochs 5 \
  --early-stopping-patience 10 \
  --checkpoint-interval 10 \
  --device mlx \
  --fp16 \
  --use-gradnorm \
  --lr-backbone 1e-5 \
  --lr-head 1e-4 \
  --compile \
  "$@"
