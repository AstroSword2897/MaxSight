# infra/

IAM policy stubs, S3 layout, encryption choices, and SageMaker environment reference for MaxSight.

## Required environment variables

| Variable | Where used | Description |
|---|---|---|
| `SAGEMAKER_ROLE_ARN` | All `scripts/ops/` launchers | Full ARN of the SageMaker execution role. See `iam/sagemaker_execution_role.json`. |
| `MAXSIGHT_S3_BUCKET` | All `scripts/ops/` launchers | S3 bucket for training artefacts, checkpoints, medallion sync, and gold uploads. |
| `AWS_DEFAULT_REGION` | `ml/infra/sagemaker_utils.py` | AWS region. Defaults to `us-east-1`. |
| `MAXSIGHT_SKIP_ROLE_ASSERT` | `ml/infra/sagemaker_utils.py` | Set to `1` to skip the account assertion in `get_execution_role`. Use only for intentional cross-account deploys. |
| `MAXSIGHT_MODEL_PACKAGE_GROUP` | `ml/infra/model_registry.py` | SageMaker Model Package Group name. When set and `register_model(..., upload_to_s3=True)` runs, a package is created with `PendingManualApproval`. |
| `MAXSIGHT_ENV` | `scripts/ops/sagemaker_deploy.py` | If `production` or `prod`, `--skip-registry-check` is **rejected** (deploy gate). |
| `SM_SUBNET_IDS` | `SMConfig` / `build_estimator`, `deploy_model` | Comma-separated subnet IDs for VPC-only training and inference. Must be used with `SM_SECURITY_GROUP_IDS`. |
| `SM_SECURITY_GROUP_IDS` | Same | Comma-separated security group IDs. |
| `SM_VOLUME_KMS_KEY_ID` | `build_estimator` | Optional KMS key id or ARN for training volume encryption at rest. |
| `MAXSIGHT_S3_PREFIX` | `SMConfig` | Key prefix inside the bucket (default `maxsight`). |

### Optional: SSM Parameter Store (recommended for CI)

Store the same values as secure parameters and export them into the job environment before calling ops scripts:

| Suggested parameter path | Maps to |
|---|---|
| `/maxsight/prod/s3_bucket` | `MAXSIGHT_S3_BUCKET` |
| `/maxsight/prod/sagemaker_role_arn` | `SAGEMAKER_ROLE_ARN` |
| `/maxsight/prod/model_package_group` | `MAXSIGHT_MODEL_PACKAGE_GROUP` |

Use **Secrets Manager** only for secrets (tokens, DB passwords); bucket name and role ARN are identifiers, not secrets, but SSM still avoids hard-coding in repo YAML.

## S3 bucket layout (gold artefacts + training outputs)

Use a single bucket (or separate buckets per env) under `MAXSIGHT_S3_PREFIX` (default `maxsight`):

| Prefix | Contents |
|---|---|
| `{prefix}/output/` | SageMaker training `model.tar.gz` and job outputs (`SMConfig.output_path`). |
| `{prefix}/checkpoints/` | Checkpoint URI base (`checkpoint_s3_uri`). |
| `{prefix}/gold/` | Gold **meta.json** and JSONL shards (upload from `build_gold_manifest` output); training YAML should reference these URIs when `data_plane: gold`. |
| `{prefix}/medallion/silver/` | Optional medallion silver layer sync (`sync_medallion_s3.py`). |
| `{prefix}/medallion/bronze/` | Optional bronze sync. |

Legacy `training_index.json` uploads from older flows may live under a path your ops team chooses; canonical new pipelines use **sharded JSONL + meta** under `gold/`.

## Encryption: SSE-S3 vs SSE-KMS

| Option | When to use |
|---|---|
| **SSE-S3 (default)** | Simpler operations; AWS manages keys; sufficient for many internal ML buckets. |
| **SSE-KMS** | Org requires CMK rotation, strict audit, or cross-account decrypt policies. Set default bucket encryption to KMS and grant the SageMaker role `kms:Decrypt` / `kms:GenerateDataKey` on that key. |

Set training volume encryption via `SM_VOLUME_KMS_KEY_ID` when the execution role is allowed to use that key for `CreateTrainingJob`.

## Versioning and lifecycle (apply in AWS)

1. **Versioning:** Turn on **S3 versioning** for the training bucket so `model.tar.gz` and gold shards can be recovered after accidental overwrites.
2. **Lifecycle:** Expire noncurrent versions and incomplete multipart uploads after N days. See `s3/bucket_lifecycle_example.json` for a template **Rules** document (adjust IDs and days; attach via AWS Console or `put-bucket-lifecycle-configuration`).

## IAM files

| File | Purpose |
|---|---|
| `iam/sagemaker_execution_role.json` | Trust policy + inline permissions for the SageMaker execution role. |
| `iam/s3_bucket_policy.json` | Bucket policy granting the execution role read/write access. |
| `iam/ecr_policy.json` | Permissions to pull AWS DLC images and push private MaxSight images. |
| `iam/kms_training_volume_policy.json` | KMS use for training volumes when `SM_VOLUME_KMS_KEY_ID` is set. |
| `iam/ssm_parameters_read_policy.json` | Read-only SSM for `/maxsight/*` (attach to CI/ops principals). |

Replace `{{ACCOUNT_ID}}`, `{{BUCKET}}`, `{{REGION}}`, `{{KMS_KEY_ID}}`, and `{{SAGEMAKER_ROLE_NAME}}` in each file before applying.

Validate JSON locally: **`python scripts/infra/validate_infra_stubs.py`**. Set `MAXSIGHT_INFRA_STRICT_PLACEHOLDERS=1` to fail on any remaining `{{…}}` except the documented KMS bucket-encryption template.

## SSM and scripts

- Parameter layout and CLI examples: **`ssm/README.md`** and **`ssm/load_env_from_ssm.example.sh`**.
- Create the SageMaker Model Package Group from the CLI: **`python scripts/ops/create_model_package_group.py --name YOUR_GROUP --dry-run`** then without `--dry-run`.

## Pre-integration checklist

See **[../docs/ops/pre_integration_checklist.md](../docs/ops/pre_integration_checklist.md)** for a single ordered gate list before deeper integration work.

## SageMaker Model Package Group (manual approval)

1. In AWS Console → SageMaker → **Model registry** → **Model package groups** → Create group (e.g. `maxsight-models-prod`), **or** run **`scripts/ops/create_model_package_group.py`**.
2. Set `MAXSIGHT_MODEL_PACKAGE_GROUP=maxsight-models-prod` in the environment used for `register_model` / training follow-up.
3. After training uploads a model to S3 and you call `register_model(..., upload_to_s3=True)`, the registry creates a package with **`PendingManualApproval`**.
4. Approve or reject in the console before treating the version as production-ready.
5. **Deploy gate:** `sagemaker_deploy.py deploy` requires the `model_data` S3 URI to exist in `runs/model_registry.json` unless `--skip-registry-check` (forbidden when `MAXSIGHT_ENV=production`).

## VPC training and inference

Set `SM_SUBNET_IDS` and `SM_SECURITY_GROUP_IDS` (both required for VPC mode). Subnets must be in subnets that can reach S3 (gateway endpoint or NAT). `build_estimator` and `deploy_model` pass them through to the SageMaker Python SDK.

**FSx for Lustre:** Attaching FSx as a training data source uses `FileSystemInput` channels; that is not wired in `build_estimator` yet. Extend the training launcher to add a channel when your org provisions FSx (see AWS *Train deep learning models using FSx for Lustre*).

## Apply order

1. Create the execution role with `sagemaker_execution_role.json` (trust policy).
2. Attach `ecr_policy.json` as an inline policy (or merge into one policy document per org standards).
3. Create the S3 bucket, enable versioning, attach lifecycle rules, set default encryption (SSE-S3 or SSE-KMS).
4. Apply `s3_bucket_policy.json` to the training bucket.
5. (Optional) Create the Model Package Group and set `MAXSIGHT_MODEL_PACKAGE_GROUP`.
6. Set `SAGEMAKER_ROLE_ARN`, `MAXSIGHT_S3_BUCKET`, and optional VPC / KMS / group env vars.
7. Run `python scripts/ops/sagemaker_train.py --dry-run ...` then a real job; see **[../docs/ops/aws_runbook.md](../docs/ops/aws_runbook.md)** for end-to-end validation.

## Account safety

`ml/infra/sagemaker_utils.py::get_execution_role` asserts that the resolved role ARN's account ID matches the caller's STS identity unless `MAXSIGHT_SKIP_ROLE_ASSERT=1`.

## CloudWatch

See **[cloudwatch/README.md](cloudwatch/README.md)** for example alarms (training job failure, endpoint 5xx). Metric definitions for training quality map from `TRAINING_METRIC_DEFINITIONS` in `ml/infra/sagemaker_utils.py` to log lines emitted by `ml/training/train_loop.py`; run **`pytest tests/test_sagemaker_integration.py`** after changing either side.
