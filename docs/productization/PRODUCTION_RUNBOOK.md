# Production and Real-World Runbook

Use this runbook to run MaxSight for production and real-world use. All paths reference the repo root.

## Prerequisites

- Python 3.9+ with dependencies from `requirements.txt` or `pyproject.toml`.
- For training: COCO (or compatible) data prepared; see `docs/downloads.md` and `docs/training-data-loading.md`.
- For export/package: a trained checkpoint (e.g. `checkpoints/<condition>/best_model.pt`).

## Canonical Commands (product pipeline)

These are the only commands needed for release and real-world deployment. See `docs/productization/03_pipeline_declutter_map.md`.

| Command | Purpose | How to run |
|--------|---------|------------|
| **train** | Train production model | `python scripts/product/run.py train --data-dir <path> [options]` or `scripts/train_maxsight.py --data-dir <path>` |
| **validate** | Model + data + safety checks | `python scripts/product/run.py validate [--checkpoint <path>] [--data]` |
| **export** | Convert checkpoint to CoreML/JIT/etc. | `python scripts/product/run.py export --checkpoint <path> --format coreml --output <path>` |
| **package** | Build glasses-ready bundle for Xcode | `python scripts/product/run.py package --checkpoint <path> --output <dir>` or `scripts/export_for_xcode.py <ckpt> <out_dir>` |
| **smoke** | Fast sanity (training + inference + export) | `python scripts/product/run.py smoke [--epochs 2]`. On CPU use `--epochs 2`; smoke may exit early on single epoch due to smoke_train early-stop. |
| **transfer** | T2 → T5 weight transfer (then fine-tune) | `python scripts/product/run.py transfer --source checkpoints/t2_hybrid_vit/best_model.pth [--config ml/training/configs/t2_to_t5_transfer.yaml]`. Then: `run.py train --resume-from <saved_init> ...`. |

## Quick production check (run before release)

From repo root:

```bash
# 1. Run full test suite
pytest tests/ -v

# 2. Run smoke (short training + inference sanity; use --epochs 2 on CPU)
python scripts/product/run.py smoke --epochs 2

# Optional: skip export validation tests if JIT trace fails in your environment (see docs/status.md)
python scripts/product/run.py validate --skip-export-tests

# 3. If you have a checkpoint: validate and export
python scripts/product/run.py validate --checkpoint checkpoints/amd/best_model.pt
python scripts/product/run.py export --checkpoint checkpoints/amd/best_model.pt --format coreml --output exports/amd.mlpackage
```

## Safety gates before release

- All mandatory gates in `docs/productization/02_safety_first_release_gates.md` must pass.
- Run `python scripts/product/run.py validate` and review any safety-gate report.
- No release without safety owner sign-off.

## Real-world deployment (glasses)

1. **Train** (or use existing checkpoint) per condition.
2. **Export** to CoreML: `python scripts/product/run.py export --checkpoint <path> --format coreml --output <path>.mlpackage`.
3. **Package** for Xcode: `scripts/export_for_xcode.py <checkpoint> <bundle_dir>`.
4. Integrate `.mlpackage` into the glasses app (see `docs/EXPORT_MODELS_TO_XCODE.md`).
5. Run pilot per `docs/productization/05_pilot_validation_protocol.md`.

## CI recommendation

- On every PR: `pytest tests/` and `python scripts/product/run.py smoke`.
- On release branch: add `scripts/product/run.py validate` and gate on safety report.

## T2 → T5 path (T5 MVP)

1. **T2 source**: Train with config that disables temporal/cross-task (e.g. `scripts/ops/train_maxsight.py --config ml/training/configs/t2_hybrid_vit.yaml --data-dir <path> --train-annotation ... --val-annotation ...`). Checkpoint goes to `checkpoints/t2_hybrid_vit/` per config.
2. **Transfer**: `python scripts/product/run.py transfer --source checkpoints/t2_hybrid_vit/best_model.pth --config ml/training/configs/t2_to_t5_transfer.yaml`. This writes an initial T5 checkpoint (e.g. `checkpoints/t5_temporal_transfer/t5_from_t2_init.pt`).
3. **T5 fine-tune**: `python scripts/product/run.py train --data-dir <path> --resume-from checkpoints/t5_temporal_transfer/t5_from_t2_init.pt ...` (optionally with video/sequence data and `t5_temporal_2phase.yaml`-style config).

## MVP runtime contract (export / app)

The shipped T5 MVP must only depend on the **MVP output keys** defined in `ml/runtime_constants.MVP_MODEL_OUTPUT_KEYS` (detections, urgency, distance, OCR, temporal consistency). The app/runtime should consume only these keys; use `ml.runtime_constants.filter_mvp_model_outputs(outputs, training=False)` when building the production inference path. Export and package use the full model; filtering is applied at runtime in the app.

## Where the plans live in the codebase

- **Product scope and claims**: `docs/productization/01_product_scope_and_claims.md`
- **Safety gates**: `docs/productization/02_safety_first_release_gates.md`
- **Pipeline declutter map**: `docs/productization/03_pipeline_declutter_map.md`
- **Runtime boundaries**: `docs/productization/04_runtime_boundary_spec.md`
- **Pilot protocol**: `docs/productization/05_pilot_validation_protocol.md`
- **This runbook**: `docs/productization/PRODUCTION_RUNBOOK.md`
- **Canonical runner**: `scripts/product/run.py`
