# Reproducibility gate (manual, before any AWS submission)

This gate exists because YAML, CLI flags, and SageMaker hyperparameters all
collide in `ResolvedTrainingConfig`. Until the gate passes for a given
config, nothing in that config may be promoted to a SageMaker training job.

## What it proves

For one tier YAML + fixed seed:

- **Run A == Run B** locally (same machine, same seed) → byte-identical
  resolved config hash, near-identical metrics through warmup.
- **Local == container** → resolved config hash matches and metrics during
  the first `warmup_epochs` are within `±epsilon` (we use the smoothness of
  loss, not absolute value, since temporal heads spike early — see notes in
  `ml/training/configs/t5_temporal.yaml`).
- **Container == SageMaker dry-run digest** → same `provenance.config_hash`
  emitted by `sagemaker_entry.py` after merging SM_HP_* env vars.

If any check fails, the resolved config is leaking through somewhere it
shouldn't (hidden default, CLI override that is not in `_OVERRIDE_MAP`, or
a SM hyperparameter without a mapping).

## Procedure

1. Pick the tier YAML you intend to ship: `CFG=ml/training/configs/t5_temporal.yaml`.
2. Capture the canonical config hash on the dev machine:
   ```bash
   python scripts/ops/train_maxsight.py --config "$CFG" --print-config \
     | python -c "import json,sys; d=json.load(sys.stdin); print(d['provenance']['config_hash'])"
   ```
3. Run A and Run B locally with the same seed (warmup epochs only is enough
   — we are gating reproducibility, not full convergence):
   ```bash
   python scripts/ops/train_maxsight.py --config "$CFG" --epochs 25 --seed 42
   python scripts/ops/train_maxsight.py --config "$CFG" --epochs 25 --seed 42
   ```
   Compare `resolved_config.json` written into `checkpoint.save_dir` — the
   `provenance.config_hash` field must match exactly.
4. Container check (Docker or `python -m`-style shim, no AWS calls):
   ```bash
   python ml/training/sagemaker_entry.py --config "$CFG" --epochs 25 --seed 42
   ```
   Confirm the printed `resolved_training_config` JSON has the same
   `config_hash`. Compare `model_meta.json["config_hash"]`.
5. SageMaker dry-run digest (no submission):
   ```bash
   python scripts/ops/sagemaker_train.py \
     --bucket dummy --config "$CFG" --epochs 25 --dry-run
   ```
   The hyperparameters block must contain only `config`, `experiment`, and
   any explicit overrides. No silent additions.

## Failure → fix flow

| Symptom | Likely cause |
|--------|--------------|
| Run A and Run B hashes differ | A non-config field flowed in (e.g. `_OVERRIDE_MAP` is missing a key, or someone added a default arg in training code). Search for the new dotted path in `ResolvedTrainingConfig` and route it through. |
| Local hash != container hash | `sagemaker_entry.py` is reading something from `os.environ` directly. Move it into `_OVERRIDE_MAP` / `_hp_overrides_from_env`. |
| Container hash != SageMaker dry-run hash | `sagemaker_train.py` passed an extra hyperparameter or omitted one the operator set. Audit the `hyperparameters` dict it builds. |
| Loss-weight mismatch | A YAML head was added without updating `_build_loss` in `ml/training/runner.py`. Schema validation should already reject it; if it doesn't, add the head to `_expected_loss_heads` in `ml/training/run_config.py`. |

## Hard rule

If the gate fails, do **not** open a SageMaker job. Fix the leak first, then
re-run the gate. The gate is the only reason local validation has any
predictive value for SageMaker results.
