# AWS runbook — one-account validation

End-to-end steps to prove training, registry, deploy, invoke, and CloudWatch metrics in a **single AWS account**. Requires CLI credentials, `SAGEMAKER_ROLE_ARN`, and `MAXSIGHT_S3_BUCKET` set.

## 1. Preconditions

- Complete **[pre_integration_checklist.md](pre_integration_checklist.md)** (ordered IAM, S3, registry, optional VPC).
- Apply IAM stubs under `infra/iam/` (placeholders replaced). See **[../../infra/README.md](../../infra/README.md)**.
- Bucket: versioning on, lifecycle optional, encryption chosen (SSE-S3 or SSE-KMS).
- Optional: `MAXSIGHT_MODEL_PACKAGE_GROUP` created in SageMaker console.
- Optional VPC: `SM_SUBNET_IDS` and `SM_SECURITY_GROUP_IDS` set if jobs must run in private subnets.

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
python scripts/ops/sagemaker_deploy.py deploy \
  --bucket "$MAXSIGHT_S3_BUCKET" \
  --role "$SAGEMAKER_ROLE_ARN" \
  --job-name YOUR_TRAINING_JOB_NAME \
  --endpoint maxsight-inference-dev
```

Do **not** use `--skip-registry-check` in production (`MAXSIGHT_ENV=production` blocks it).

## 6. Invoke

```bash
python scripts/ops/sagemaker_deploy.py invoke \
  --endpoint maxsight-inference-dev \
  --region "$AWS_DEFAULT_REGION" \
  --payload '{"ping":true}'
```

Then with a small test image:

```bash
python scripts/ops/sagemaker_deploy.py invoke \
  --endpoint maxsight-inference-dev \
  --region "$AWS_DEFAULT_REGION" \
  --image tests/fixtures/test_frame.jpg
```

## 7. CloudWatch metrics

- Open the training job → **Monitor** → **View logs** → confirm log lines include `Train Loss:`, `Val mAP:`, etc.
- In CloudWatch **Metrics**, search under custom namespace if your account surfaces SageMaker algorithm metrics; otherwise rely on log-based metric definitions from `TRAINING_METRIC_DEFINITIONS`.
- Run locally: `pytest tests/test_sagemaker_integration.py` to lock regex ↔ log contract.

## 8. CI

GitHub Actions job **`sagemaker_ops`** in `.github/workflows/ci.yml` runs ops dry-run and SageMaker-related tests without submitting jobs.
