# Project status and health

This document summarizes the current status of the MaxSight repo: what is implemented, what is tested, and what to watch when running or deploying.

## Implementation status

- **Model (MaxSightCNN):** Implemented in `ml/models/maxsight_cnn.py` with tiered capabilities (T0–T5). Full wiring is ~393M parameters; CI size bounds are in `ml/runtime_constants.py`.
- **Training:** Production loop with checkpoint resume, skipped-batch guards, and `health_summary` logs in `ml/training/train_loop.py`. Observability contracts in `ml/training/observability.py`. Ops scripts under `scripts/ops/`.
- **Export:** JIT, ExecuTorch (.pte), CoreML, and ONNX export in `ml/training/export.py`.
- **Therapy:** Closed-loop engine with `update()` + `on_user_response()` in `ml/therapy/`. Simulator integration in `tools/simulation/web_simulator.py`. See `docs/therapy_system.md` and `docs/ops/production_remediation.md`.
- **Haptics:** Platform backends in `app/ui/haptic_backends.py`; facade in `app/ui/haptic_feedback.py`.
- **Retrieval:** Async, advisory-only enhancement in `ml/retrieval/`.
- **Simulation:** Web simulator under `tools/simulation/`.

## Tests

- **Production remediation:** `tests/test_production_remediation.py` — checkpoint resume, therapy closed loop, SageMaker channels, haptics.
- **CI contracts:** `python scripts/infra/validate_train_loop_contracts.py` (also in `.github/workflows/ci.yml`).
- **Phase and integration:** `pytest tests/` — model, data, therapy, SageMaker, ops launchers.
- **Benchmark:** `python -m ml.training.benchmark`

## Known limitations and risks

- **Model size:** Full MaxSightCNN is ~375 MB INT8 (~393M params). Tiered or quantized deployment may use smaller checkpoints.
- **Haptics:** macOS requires PyObjC or Swift; Linux requires `evdev` and FF-capable hardware. Use `log` backend for dev/CI.
- **JIT / ExecuTorch export:** Tracing can hit unsupported ops; use CPU + JIT-only for stability.
- **Dataset lighting flags:** Use `tag_lighting_metadata` and `lighting_pixel_augmentation` (not legacy `apply_lighting_augmentation`).

## Branch and deployment

- **Default branch:** Check your current branch (e.g. `feature/multimodal_refactor`). Main deployment and export instructions in README apply to the state on that branch.
- **CI:** `.github/workflows/ci.yml` runs lint and tests on push/PR. Green CI indicates tests and lint pass for that commit.

## How to use this doc

- **Before training:** Ensure data is prepared (see `docs/downloads.md`, `docs/training-data-loading.md`) and that you have the right annotation and image paths.
- **Before export:** Run a quick validation (e.g. `scripts/deploy_top7.py --validate-only`) and use CPU + JIT-only if you see crashes.
- **After changes:** Run `pytest tests/` and, if relevant, `python -m ml.training.benchmark` to confirm nothing regressed.

For deeper technical detail, see `docs/architecture.md`, `docs/training_architecture.md`, and `docs/therapy_system.md`.
