# Config selection and mAP target

## Target: mAP@0.5 ≥ 0.5 (inference only, no retraining)

- Improve mAP by sweeping confidence and NMS on **existing** checkpoints only. No retraining.
- Run: `python scripts/improve_map_all_models.py --checkpoints-base <path> --confidence 0.05 --nms-iou 0.5` (or omit `--skip-sweep` to sweep for best params).
- Configs below are for reference (T5 architecture and loss balance). Do not retrain; use current checkpoints.

## Which config to use

| Config | Use case | mAP target |
|--------|----------|------------|
| **t5_temporal.yaml** | Full T5 reference, ~320M params | 0.5 |
| **t5_sec.yaml** | T5 safety-focused loss balance reference | 0.5 |
| t0_baseline through t4 | Legacy tiers; repo is T5-only | N/A |
