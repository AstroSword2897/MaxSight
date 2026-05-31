#!/usr/bin/env bash
# Example: export MaxSight ops env vars from SSM before train/deploy.
# Copy to load_env_from_ssm.sh, set PREFIX if needed, then: source ./load_env_from_ssm.sh
set -euo pipefail
PREFIX="${MAXSIGHT_SSM_PREFIX:-/maxsight/prod}"
REGION="${AWS_DEFAULT_REGION:-us-east-1}"

read_param() {
  aws ssm get-parameter --name "${PREFIX}/$1" --region "$REGION" --query Parameter.Value --output text
}

export MAXSIGHT_S3_BUCKET="$(read_param s3_bucket)"
export SAGEMAKER_ROLE_ARN="$(read_param sagemaker_role_arn)"
# Optional: create /maxsight/prod/model_package_group in SSM when using SageMaker Model Registry.
if aws ssm get-parameter --name "${PREFIX}/model_package_group" --region "$REGION" &>/dev/null; then
  export MAXSIGHT_MODEL_PACKAGE_GROUP="$(read_param model_package_group)"
fi
# Optional VPC (uncomment parameter names in SSM to match):
# export SM_SUBNET_IDS="$(read_param sm_subnet_ids)"
# export SM_SECURITY_GROUP_IDS="$(read_param sm_security_group_ids)"
