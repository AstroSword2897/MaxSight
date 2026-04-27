# infra/

IAM policy stubs and infrastructure reference for the MaxSight AWS deployment.

## Required environment variables

| Variable | Where used | Description |
|---|---|---|
| `SAGEMAKER_ROLE_ARN` | All `scripts/ops/` launchers | Full ARN of the SageMaker execution role. See `iam/sagemaker_execution_role.json`. |
| `MAXSIGHT_S3_BUCKET` | All `scripts/ops/` launchers | S3 bucket for training artifacts, checkpoints, and medallion layers. |
| `AWS_DEFAULT_REGION` | `ml/infra/sagemaker_utils.py` | AWS region. Defaults to `us-east-1`. |
| `MAXSIGHT_SKIP_ROLE_ASSERT` | `ml/infra/sagemaker_utils.py` | Set to `1` to skip the account/region assertion in `get_execution_role`. Use only for intentional cross-account deploys. |

## IAM files

| File | Purpose |
|---|---|
| `iam/sagemaker_execution_role.json` | Trust policy + inline permissions for the SageMaker execution role. |
| `iam/s3_bucket_policy.json` | Bucket policy granting the execution role read/write access. |
| `iam/ecr_policy.json` | Permissions to pull AWS DLC images and push private MaxSight images. |

Replace `{{ACCOUNT_ID}}`, `{{BUCKET}}`, `{{REGION}}`, and `{{SAGEMAKER_ROLE_NAME}}` in each file before applying.

## Apply order

1. Create the execution role with `sagemaker_execution_role.json` (trust policy).
2. Attach `ecr_policy.json` as an inline policy.
3. Apply `s3_bucket_policy.json` to your training bucket.
4. Set `SAGEMAKER_ROLE_ARN` to the resulting role ARN.
5. Run `python scripts/ops/sagemaker_train.py --dry-run` to verify config before submitting.

## Account safety

`ml/infra/sagemaker_utils.py::get_execution_role` asserts that the resolved role ARN's
account ID matches the caller's STS identity. This prevents wrong-account deploys
caused by stale environment variables or copy-paste errors in `SAGEMAKER_ROLE_ARN`.
