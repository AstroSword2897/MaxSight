# Export top 7 models and send to Xcode

This guide exports the top 7 condition-specific models (by mAP or fixed list) to iOS bundles and prepares them for Xcode.

## Prerequisites

- Checkpoints in the expected layout: `<checkpoints-base>/checkpoints_<condition>/best_model.pt`
- Optional: `inference_data.json` from `run_checkpoint_inference` or `improve_map_all_models` if you want the **top 7 by mAP** instead of the fixed list.

## One-shot: deploy top 7 to iOS bundles

From the repo root:

```bash
# Use auto-detected checkpoints base (or set CHECKPOINTS_BASE).
python scripts/deploy_top7.py --output-dir exports/top7

# Or specify base and output.
python scripts/deploy_top7.py --checkpoints-base /path/to/MaxSight --output-dir /path/to/MaxSight/exports_top7
```

- Each condition is exported to `--output-dir/<condition>/` (e.g. `exports/top7/amblyopia/`).
- Each folder contains an **Xcode-ready bundle**: traced model (JIT or PTE), `model_config.json`, `runtime_config.json`, `processing_reference.py`, and `README_XCODE.md`.
- `manifest.json` at the output root lists all conditions and paths.

## Top 7 by mAP (rank by validation mAP)

If you have run inference and have `inference_data.json`:

```bash
python scripts/deploy_top7.py --checkpoints-base /path/to/MaxSight --output-dir exports/top7 --top-by-map --inference-data inference_data.json
```

Conditions are then chosen by highest mAP@0.5 instead of the fixed list.

## Full pipeline: inference + deploy

To run inference (to build or refresh `inference_data.json`) and then deploy the top 7:

```bash
python scripts/inference_and_deploy_top7.py \
  --checkpoints-base /path/to/MaxSight \
  --output-dir /path/to/MaxSight/exports_top7 \
  --val-annotation /path/to/cleaned_splits/maxsight_val.json \
  --image-dir /path/to/data \
  --max-batches 10
```

Use `--top-by-map` to deploy the top 7 by mAP after inference.

## Validate only (no export)

Check that all seven checkpoints load and run one-batch inference:

```bash
python scripts/deploy_top7.py --checkpoints-base /path/to/MaxSight --output-dir exports/top7 --validate-only
```

## Sending bundles to Xcode

1. Copy each condition folder (e.g. `exports/top7/amblyopia/`) into your Xcode project or a shared resource location.
2. In each folder, open `README_XCODE.md` for steps to add the model and configs to your app target.
3. Port preprocessing from `processing_reference.py` to Swift using the same normalization and input size (e.g. 224x224).

## Single-condition export (one model to Xcode)

For one checkpoint only:

```bash
python scripts/export_for_xcode.py checkpoints_amblyopia/best_model.pt maxsight_ios_bundle
```

Or use `export_one_model.py` for JIT-only:

```bash
python scripts/export_one_model.py --checkpoint /path/to/checkpoints_amblyopia/best_model.pt --out maxsight.pt --device cpu --no-subprocess
```

## Troubleshooting

- **`/bin/bash: line 1: path: No such file or directory` (e.g. after "Already up to date"):** Something is running the literal command `path`. On macOS/Linux there is no `path` command. Check Cursor/VS Code: **Settings** → search for "git" or "sync" or "pull" → look for a "run after pull/sync" or "post pull command" set to `path` and clear it or set it to a real command (e.g. `echo "Sync done"`). Do not run the word `path` from doc examples; use real paths (e.g. `.` or `$PWD`).
- **No checkpoints found:** Ensure `<base>/checkpoints_<cond>/best_model.pt` exists. Run `python scripts/find_trained_checkpoints.py` to discover paths.
- **Export segfault (exit 139):** JIT trace can crash on some environments; the export pipeline stubs the CLIP encoder to reduce this. Use `--device cpu` and `--no-subprocess` to see a full traceback.
- **JIT-only vs PTE:** Default is JIT-only (`--quick`). Use `--no-quick` to try ExecuTorch PTE first (requires ExecuTorch installed).
