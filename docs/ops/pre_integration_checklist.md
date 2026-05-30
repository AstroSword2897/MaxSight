# Pre-integration checklist (AWS / SageMaker)

Complete these steps before wiring additional systems (CI secrets, multi-account, or app-tier inference). Order matches **[../../infra/README.md](../../infra/README.md)**.

## 1. IAM and roles

- [ ] Replace placeholders in `infra/iam/sagemaker_execution_role.json`, `s3_bucket_policy.json`, and `ecr_policy.json`.
- [ ] If using `SM_VOLUME_KMS_KEY_ID`, merge `infra/iam/kms_training_volume_policy.json` into the execution role.
- [ ] For SSM-driven CI or operators, attach `infra/iam/ssm_parameters_read_policy.json` to the **CI/Ops role** (not necessarily the SageMaker execution role).
- [ ] Create the SageMaker execution role and note `SAGEMAKER_ROLE_ARN`.

## 2. S3 bucket

- [ ] Create the artefacts bucket; set default encryption (**`infra/s3/bucket_encryption_sse_s3.json`** or **`bucket_encryption_sse_kms.json`**).
- [ ] Enable versioning; attach lifecycle rules (**`infra/s3/bucket_lifecycle_example.json`**).
- [ ] Apply bucket policy **`infra/iam/s3_bucket_policy.json`**.

## 3. SageMaker registry and config

- [ ] Create a Model Package Group (console or **`python scripts/ops/create_model_package_group.py --name ...`**).
- [ ] Set `MAXSIGHT_MODEL_PACKAGE_GROUP` (and optionally store it in SSM per **`infra/ssm/README.md`**).
- [ ] Confirm **`python scripts/infra/validate_infra_stubs.py`** exits 0 locally after editing JSON stubs.

## 4. Network (optional)

- [ ] If jobs run in VPC: set `SM_SUBNET_IDS` and `SM_SECURITY_GROUP_IDS`; confirm S3 access (endpoint or NAT).
- [ ] Run **`python scripts/ops/sagemaker_train.py ... --dry-run`** and **`python scripts/ops/sagemaker_deploy.py deploy ... --dry-run`** (with a registered artifact or `--skip-registry-check` in non-prod) and confirm optional `vpc` in printed JSON.

## 5. Validation in one account

- [ ] Follow **[aws_runbook.md](aws_runbook.md)** through invoke; confirm training logs contain metric lines matching `TRAINING_METRIC_DEFINITIONS` in `ml/infra/sagemaker_utils.py`.

## 6. CI and observability

- [ ] Mirror required env vars in GitHub Actions secrets or SSM-backed workflow steps (never commit real ARNs).
- [ ] Wire CloudWatch: see **`infra/cloudwatch/README.md`** and **`infra/cloudwatch/eventbridge_training_job_failed.json`** for training-failure signal.
- [ ] Ensure production deploys use `MAXSIGHT_ENV=production` so **`--skip-registry-check`** cannot bypass the registry gate.
