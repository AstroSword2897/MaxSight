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

## Where the plans live in the codebase

- **Product scope and claims**: `docs/productization/01_product_scope_and_claims.md`
- **Safety gates**: `docs/productization/02_safety_first_release_gates.md`
- **Pipeline declutter map**: `docs/productization/03_pipeline_declutter_map.md`
- **Runtime boundaries**: `docs/productization/04_runtime_boundary_spec.md`
- **Pilot protocol**: `docs/productization/05_pilot_validation_protocol.md`
- **This runbook**: `docs/productization/PRODUCTION_RUNBOOK.md`
- **Canonical runner**: `scripts/product/run.py`
