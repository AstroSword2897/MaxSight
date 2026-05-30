# SSM Parameter Store (MaxSight)

Use **Standard** parameters for non-secret identifiers the repo already documents in **[../README.md](../README.md)** (`infra/README.md` env table).

## Suggested paths

| Parameter name | Typical type | Maps to |
|---|---|---|
| `/maxsight/prod/s3_bucket` | String | `MAXSIGHT_S3_BUCKET` |
| `/maxsight/prod/sagemaker_role_arn` | String | `SAGEMAKER_ROLE_ARN` |
| `/maxsight/prod/model_package_group` | String | `MAXSIGHT_MODEL_PACKAGE_GROUP` |
| `/maxsight/prod/s3_prefix` | String | `MAXSIGHT_S3_PREFIX` |
| `/maxsight/prod/sm_subnet_ids` | String | `SM_SUBNET_IDS` (comma-separated) |
| `/maxsight/prod/sm_security_group_ids` | String | `SM_SECURITY_GROUP_IDS` |

## Bootstrap examples

Put parameters (replace account-specific values):

```bash
aws ssm put-parameter --name /maxsight/prod/s3_bucket --value my-artifacts-bucket --type String --overwrite
aws ssm put-parameter --name /maxsight/prod/sagemaker_role_arn --value arn:aws:iam::123456789012:role/MaxSightSageMakerRole --type String --overwrite
```

Load into the shell before ops scripts (see **`load_env_from_ssm.example.sh`**).

## Secrets Manager

Store API tokens and database credentials in **Secrets Manager**, not SSM String parameters. Grant read only to the runtime role that needs them; keep SageMaker execution roles scoped to training artefacts unless a job explicitly assumes a secret-reader role.
