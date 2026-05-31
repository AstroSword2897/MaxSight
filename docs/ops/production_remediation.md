# Production remediation (2026-05)

This document is the canonical reference for the May 2026 production remediation pass: training reliability, therapy closed-loop behavior, haptic backends, SageMaker contracts, and CI observability gates.

## Summary

| Area | Change | Primary files |
|------|--------|---------------|
| Training loop | Checkpoint resume raises on corrupt files; skipped-batch ratio guard; `health_summary` epoch logs; best checkpoint aligned to `best_model.pt` | `ml/training/train_loop.py`, `ml/training/observability.py` |
| SageMaker | Channel remapping (`SM_CHANNEL_TRAIN` / `SM_CHANNEL_VAL` → config paths) | `ml/training/sagemaker_entry.py` |
| Therapy | Closed-loop `on_user_response()` wired in simulator; safety/rate limits; docstrings | `ml/therapy/*`, `tools/simulation/web_simulator.py` |
| Haptics | Platform backends replace stub gating | `app/ui/haptic_backends.py`, `app/ui/haptic_feedback.py` |
| Personalization | Deterministic user hash; adapted fusion weights applied | `app/personal_mode.py` |
| CI | Contract validator + integration tests | `scripts/infra/validate_train_loop_contracts.py`, `tests/test_production_remediation.py` |

---

## Training loop contracts

### Checkpoints

- **Best checkpoint filename:** `best_model.pt` (local loop and SageMaker entrypoint).
- **Resume behavior:**
  - Missing checkpoint path → `checkpoint_resume_status = "missing"`, training starts fresh.
  - Corrupt or incompatible checkpoint → **raises** (does not silently continue).
- **Composite best model:** Early stopping considers validation loss and optional val mAP.

### Skipped-batch ratio

`ProductionTrainLoop` tracks `skipped_batches` per epoch. When the ratio exceeds `max_skipped_batch_ratio` (default **0.1**, from `ml/training/observability.py`), training aborts with `RuntimeError`.

```python
from ml.training.observability import DEFAULT_MAX_SKIPPED_BATCH_RATIO, validate_skipped_batch_ratio
```

### Health summary logs

Each epoch emits a structured log line prefixed with `health_summary`:

```
health_summary epoch=3 processed_batches=12 skipped_batches=2 skip_ratio=14.29% train_loss=1.2345 val_loss=2.3456 val_map=0.4567 new_best=True lr=1.000000e-03
```

CloudWatch metric definitions in `ml/infra/sagemaker_utils.py` parse these lines for:

- `train:processed_batches`
- `train:skipped_batches`
- `train:skip_ratio_pct`

### CI validation

Run locally:

```bash
python scripts/infra/validate_train_loop_contracts.py
pytest tests/test_production_remediation.py tests/test_sagemaker_integration.py -v
```

Both run in `.github/workflows/ci.yml` (`test` and `sagemaker_ops` jobs).

---

## SageMaker channel remapping

`apply_sagemaker_channel_paths()` in `ml/training/sagemaker_entry.py` rewrites `ResolvedTrainingConfig.data.*` paths to SageMaker channel mount directories:

| Env var | Role |
|---------|------|
| `SM_CHANNEL_TRAIN` | Training data channel (required) |
| `SM_CHANNEL_VAL` | Validation data channel (required) |
| `SM_CHANNEL_GOLD` | Optional gold index channel |

The entrypoint asserts train and val channels exist before calling `run_training()`.

---

## Therapy closed loop

### Pipeline

```
Perception → SituationUnderstanding → TherapyDecisionEngine → InterventionGenerator
  → CrossModalScheduler (voice/haptic) → User response (next frame)
  → ResponseEvaluation → AdaptationEngine → TherapyMemory
```

### Key APIs

- **`TherapyEngine.update(perception)`** — Returns `List[TherapeuticAction]` for delivery.
- **`TherapyEngine.on_user_response(perception_after)`** — Evaluates prior intervention effectiveness and updates adaptation memory.

The web simulator sets `_awaiting_therapy_response` after delivering an action and calls `on_user_response()` on the next perception tick when the user has had time to respond.

### Safety layer

`TherapySafetyLayer` suppresses prompts when:

- Perception uncertainty exceeds `THERAPY_UNCERTAINTY_SUPPRESS_THRESHOLD` (0.7)
- Minimum gap between prompts not elapsed
- Max prompts per minute exceeded

Content is sanitized to block medical/diagnostic phrasing.

---

## Haptic backends

### Configuration

| Variable | Values | Description |
|----------|--------|-------------|
| `MAXSIGHT_HAPTIC_BACKEND` | `auto`, `darwin`, `linux`, `log`, `none` | Backend selection |
| `MAXSIGHT_ENABLE_HAPTICS_STUB` | `0` / `1` | Simulator: when `1`, forces `log` backend |

### Platform support

| Backend | Platform | Requirements |
|---------|----------|--------------|
| `DarwinHapticBackend` | macOS | PyObjC (`AppKit`) or `swift` on PATH |
| `LinuxEvdevHapticBackend` | Linux | `evdev` package + FF-capable input device |
| `LogHapticBackend` | Any | Structured logs only (dev/CI) |
| `NoopHapticBackend` | Any | Silent no-op |

### Usage

```python
from app.ui.haptic_feedback import HapticFeedback, HapticPattern

haptics = HapticFeedback(backend="auto", allow_log_fallback=True)
haptics.trigger(HapticPattern.MICRO_PULSE, intensity=0.5)
haptics.stop()
```

---

## Model size CI envelope

Full MaxSightCNN wiring is ~393M parameters (~375 MB INT8). CI bounds live in `ml/runtime_constants.py`:

| Constant | Value |
|----------|-------|
| `DEFAULT_MODEL_MIN_PARAMS` | 90,000,000 |
| `DEFAULT_MODEL_MAX_PARAMS` | 400,000,000 |
| `DEFAULT_MODEL_INT8_MAX_MB` | 400.0 |

Tests in `tests/test_model.py`, `tests/test_comprehensive_system.py`, and `tests/test_performance.py` import these constants.

---

## Dataset lighting parameters

`MaxSightDataset` renamed the legacy `apply_lighting_augmentation` flag into two explicit knobs:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `tag_lighting_metadata` | `True` | Attach dim/normal lighting tag to sample dict (no pixel change) |
| `lighting_pixel_augmentation` | `False` | Apply luminance/gamma pixel transforms when implemented |

Use these names in new code and tests. The old name is deprecated.

---

## Integration test coverage

`tests/test_production_remediation.py`:

1. Checkpoint resume — missing, corrupt, init-with-corrupt-path
2. Skipped-batch ratio contract
3. Health summary parser roundtrip
4. Therapy closed-loop `on_user_response()`
5. Simulator therapy response flow (unit)
6. SageMaker channel remapping (unit + entrypoint)
7. Haptic backend resolution and injected backend

---

## Related docs

- [AWS runbook](aws_runbook.md)
- [Pre-integration checklist](pre_integration_checklist.md)
- [Training data loading](../training-data-loading.md)
- [Therapy system](../therapy_system.md)
- [Project status](../status.md)
