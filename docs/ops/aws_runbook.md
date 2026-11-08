# AWS runbook — one-account validation

End-to-end steps to prove training, registry, deploy, invoke, and CloudWatch metrics in a **single AWS account**. Requires CLI credentials, `SAGEMAKER_ROLE_ARN`, and `MAXSIGHT_S3_BUCKET` set.

## 1. Preconditions

- Complete **[pre_integration_checklist.md](pre_integration_checklist.md)** (ordered IAM, S3, registry, optional VPC).
- Apply IAM stubs under `infra/iam/` (placeholders replaced). See **[../../infra/README.md](../../infra/README.md)**.
- Bucket: versioning on, lifecycle optional, encryption chosen (SSE-S3 or SSE-KMS).
- Optional: `MAXSIGHT_MODEL_PACKAGE_GROUP` created in SageMaker console.
- Optional VPC: `SM_SUBNET_IDS` and `SM_SECURITY_GROUP_IDS` set if jobs must run in private subnets.
- **Full test suite in AWS (prep gate):** on an instance with ≥ 10 GB free disk, install `requirements.txt`
  (includes `faiss-cpu`, `cryptography`) and run `pytest -q`. The last clean local baseline was 1301 passed,
  5 skipped, 12 xfailed, 0 failed; the 2026-11-07 revision was verified locally with targeted suites only,
  because the local disk filled up. Checklist:
  [../architecture/MAXSIGHT_ARCHITECTURE.md §15](../architecture/MAXSIGHT_ARCHITECTURE.md#15-aws-preparation-checklist).
- **Model packages:** `model.tar.gz` must contain `best.pt` and `model_meta.json` (with `tier`). The
  endpoint refuses to start without `best.pt` or with a checkpoint that does not match the model exactly
  (it no longer serves random weights).
- **Signing (releases for devices):** store the Ed25519 private key in Secrets Manager, mount it as a file,
  and set `MAXSIGHT_SIGNING_KEY_FILE`; sign with `scripts/product/run.py certify --sign --artifact <model.pt>`.
  Never put the key in the repository or an image. Provisioning: `ml/runtime/stage_a/keys/README.md`.

## 2. Dry-run (no AWS mutation beyond optional S3 upload in train script)

```bash
export MAXSIGHT_S3_BUCKET=your-bucket
export SAGEMAKER_ROLE_ARN=arn:aws:iam::ACCOUNT:role/YourSageMakerRole

python scripts/ops/sagemaker_train.py \
  --bucket "$MAXSIGHT_S3_BUCKET" \
  --role "$SAGEMAKER_ROLE_ARN" \
  --config ml/training/configs/t0_baseline.yaml \
  --dry-run
```

Confirm printed JSON includes `output_path`, `gold_s3` (if medallion index exists locally), and optional `vpc` / `volume_kms_key`.

## 3. Submit a short real training job

Remove `--dry-run`. Use a small tier config and few epochs via hyperparameters if the YAML allows overrides.

```bash
python scripts/ops/sagemaker_train.py \
  --bucket "$MAXSIGHT_S3_BUCKET" \
  --role "$SAGEMAKER_ROLE_ARN" \
  --config ml/training/configs/t0_baseline.yaml \
  --epochs 1
```

Wait for **Succeeded** in the SageMaker console. Note `TrainingJobName`.

## 4. Register artefact (local registry + optional Model Package)

- Copy `model.tar.gz` S3 URI from the job output (or use `get_latest_training_job_output` logic).
- Register in `runs/model_registry.json` via your process, or use `sagemaker_deploy.py --register` after first deploy.
- If `MAXSIGHT_MODEL_PACKAGE_GROUP` is set and you use `ModelRegistry.register_model(..., upload_to_s3=True)`, approve the package in the console when status is **Pending manual approval**.

## 5. Deploy endpoint

```bash
python scripts/ops/sagemaker_deploy.py \
  --bucket "$MAXSIGHT_S3_BUCKET" \
  --role "$SAGEMAKER_ROLE_ARN" \
  deploy \
  --job-name YOUR_TRAINING_JOB_NAME \
  --endpoint maxsight-inference-dev
```

Do **not** use `--skip-registry-check` in production (`MAXSIGHT_ENV=production` blocks it).

## 6. Invoke

```bash
python scripts/ops/sagemaker_deploy.py \
  --region "$AWS_DEFAULT_REGION" \
  invoke \
  --endpoint maxsight-inference-dev \
  --payload '{"ping":true}'
```

Then with a small test image:

```bash
python scripts/ops/sagemaker_deploy.py \
  --region "$AWS_DEFAULT_REGION" \
  invoke \
  --endpoint maxsight-inference-dev \
  --image tests/fixtures/test_frame.jpg
```

## 7. CloudWatch metrics

- Open the training job → **Monitor** → **View logs** → confirm log lines include `Train Loss:`, `Val mAP:`, etc.
- In CloudWatch **Metrics**, search under custom namespace if your account surfaces SageMaker algorithm metrics; otherwise rely on log-based metric definitions from `TRAINING_METRIC_DEFINITIONS`.
- Run locally: `pytest tests/test_sagemaker_integration.py` to lock regex ↔ log contract.

## 8. CI

GitHub Actions job **`sagemaker_ops`** in `.github/workflows/ci.yml` runs ops dry-run and SageMaker-related tests without submitting jobs.
