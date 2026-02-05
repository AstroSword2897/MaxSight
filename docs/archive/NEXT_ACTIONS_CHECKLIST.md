# Next Actions Checklist — M3 Pro / MPS / GradNorm / CoreML / Rehearsal

Use this checklist to close the loop on M3 Pro dev readiness and production rehearsal. Share with the team as-is or copy sections into your tracking tool.

---

## 1. MPS / Latency

- [ ] On an M3 Pro (or Apple Silicon) machine, run benchmark scripts to confirm sync gives **consistent latency**:
  - `python scripts/archive/validate_forward_passes.py` (with MPS if available)
  - `python scripts/archive/benchmark_tiers.py` (tier of choice; MPS device)
- [ ] Note: first-run warmup may show minor variance; a few runs should stabilize.

---

## 2. GradNorm (MPS)

- [ ] Optionally run a **short training run on MPS with GradNorm** to confirm CPU fallback:
  - `python scripts/train_maxsight.py --device mps --use-gradnorm ...` (minimal epochs/steps).
- [ ] Confirm no MPS-specific errors; task weights should update and be pushed back to device.
- [ ] For “no GradNorm” runs, omit `--use-gradnorm` (default is off).

---

## 3. CoreML Multi-Input (if needed)

- [ ] **Decide** whether audio/temporal inputs are required for CoreML export.
- [ ] If **yes**: extend `ml/training/export.py` → `export_to_coreml()` to add `audio_features` (and any temporal inputs) with **fixed shapes** (e.g. `(1, 128)`), then run a test export.
- [ ] If **no**: document that CoreML export is **image-only** and close as known limitation.
- [ ] Run CoreML export test on M3 Pro when ready: e.g. `pytest tests/test_export_validation.py` or the CoreML step in `scripts/archive/benchmark_tiers.py`.

---

## 4. Production Rehearsal

- [ ] Run full rehearsal and inspect logs:
  ```bash
  python scripts/archive/full_production_rehearsal.py --device mps --num-frames 5 --log-dir logs
  ```
  (Use `--device cpu` if MPS not available.)
- [ ] Check `logs/production_rehearsal.log` and `logs/production_rehearsal_results.json` for:
  - Skipped frames or pipeline breakdowns above threshold.
  - Alert counts vs `max_alerts_per_frame` (over-limit or misfires).
  - Any errors per scenario (rain, glare, tilt, combined).
- [ ] Tune `max_alerts_per_frame` / `alert_cooldown_frames` in `tools/simulation/config.py` if needed.

---

## 5. Final Report / CI

- [ ] Run full test suite for regressions:
  ```bash
  pytest tests/test_production_hardening.py tests/test_critical_fixes.py tests/test_model.py tests/test_gradnorm_integration.py tests/test_export_validation.py -v
  ```
- [ ] Update docs: confirm **M3 Pro dev readiness** and any **CoreML multi-input limitation** are clearly stated (e.g. in `README.md`, `REQUIREMENTS.md`, or `docs/DEPLOYMENT_BOTTLENECKS.md`).
- [ ] Optional: add a CI job or nightly that runs `scripts/archive/full_production_rehearsal.py` and/or MPS benchmarks.

---

## 6. Data requirements (gathering and arm64)

- [ ] **Gather all data** so training and AutoML use the same layout (works on x86_64 and arm64):
  ```bash
  python scripts/gather_training_data.py [--data-dir datasets/coco_raw] [--splits-dir datasets/cleaned_splits] [--skip-download] [--skip-extract] [--download-auto]
  ```
- [ ] Then train with annotation-based layout:
  ```bash
  python scripts/train_maxsight.py --data-dir datasets/coco_raw --train-annotation datasets/cleaned_splits/maxsight_train.json --val-annotation datasets/cleaned_splits/maxsight_val.json --image-dir datasets/coco_raw --epochs 2 --device cpu
  ```
- [ ] On arm64 use `--device cpu` if MPS has unsupported ops; use `--device mps` when supported. See `REQUIREMENTS.md` for data and AutoML layout.

---

## 7. Hyperparameter tuning (AutoML)

- [ ] Run Optuna-based tuning when you want to search over learning rate, weight decay, batch size, and gradient clip instead of hand-picking:
  ```bash
  python scripts/tune_hyperparameters.py --data-dir /path/to/data --n-trials 20 --epochs-per-trial 5 --use-fp16
  ```
- The script uses **Optuna** and **minimizes validation loss**. Best hyperparameters are written to `checkpoint_dir/best_hyperparameters.json` (default: `./checkpoints_tuning/best_hyperparameters.json`). The best trial’s `best_model.pt` is copied to `checkpoint_dir/best_model.pt`.
- For **full training** after tuning, run `scripts/train_maxsight.py` with the suggested params (e.g. by passing the JSON values manually or parsing `best_hyperparameters.json`).

---

## Quick reference

| Area        | Status in code | Your action |
|------------|----------------|-------------|
| MPS sync   | Done           | Run benchmarks; confirm stable latency. |
| GradNorm   | Done (CPU fallback on MPS) | Optional: short MPS train with `--use-gradnorm`. |
| CoreML     | Image-only     | Add multi-input if needed; else document limitation. |
| Rehearsal  | Implemented    | Run script; review logs for thresholds/alerts. |
| Tests      | Passing (as of last run) | Re-run pytest before closing. |
| Data gathering | gather_training_data.py | One-time: download/extract COCO, create splits; then use --train-annotation/--val-annotation. |
| Hyperparameter tuning | Optuna script | Run `tune_hyperparameters.py`; use `best_hyperparameters.json` for full training. |
