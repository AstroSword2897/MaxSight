# Config selection and mAP target

## Target: mAP@0.5 ≥ 0.5

- Use **t5_temporal.yaml** for full T5 training (temporal + hybrid + cross-task + cross-modal). Trained weights are required; minimal/placeholder checkpoints give mAP ~0.002.
- After training, run inference and sweep to improve mAP: `python scripts/improve_map_all_models.py --checkpoints-base <path> --confidence 0.05 --nms-iou 0.5` (or use `--skip-sweep` with fixed params).
- To reach mAP 0.5: train with t5_temporal (or t5_sec for safety-focused) for the configured epochs, then run the improve-mAP pipeline on the resulting checkpoints.

## Which config to use

| Config | Use case | mAP target |
|--------|----------|------------|
| **t5_temporal.yaml** | Full T5, cloud GPU, ~320M params | 0.5 |
| **t5_sec.yaml** | T5 safety path, same arch with safety-focused loss balance | 0.5 |
| t0_baseline through t4 | Legacy tiers; repo is T5-only | N/A |
