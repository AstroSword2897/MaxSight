"""SageMaker session, role, and job-configuration helpers.

Abstracts the SageMaker SDK so the rest of the codebase can reference
``get_session()`` / ``get_execution_role()`` without touching boto3 directly.

IAM reference: see ``infra/iam/sagemaker_execution_role.json`` for the trust
policy and permission boundaries required by this role. Set SAGEMAKER_ROLE_ARN
to the resolved ARN before running any ``scripts/ops/`` launcher.

Usage
-----
from ml.infra.sagemaker_utils import get_session, get_execution_role, SMConfig

cfg = SMConfig.from_env()
session = get_session(cfg.region)
role = get_execution_role(cfg.role_name_or_arn)
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from ml.training.observability import cloudwatch_health_metric_definitions

# CloudWatch metrics parsed from ProductionTrainLoop logs (train_loop.py).
TRAINING_METRIC_DEFINITIONS = [
    {"Name": "train:loss", "Regex": r"Train Loss: ([0-9.eE+-]+)"},
    {"Name": "val:loss", "Regex": r"Val Loss: ([0-9.eE+-]+)"},
    {"Name": "val:mAP", "Regex": r"Val mAP: ([0-9.eE+-]+)"},
    {"Name": "val:mAP50", "Regex": r"Val mAP@0.5: ([0-9.eE+-]+)"},
    {"Name": "val:mAP75", "Regex": r"Val mAP@0.75: ([0-9.eE+-]+)"},
    {"Name": "val:precision", "Regex": r"Precision: ([0-9.eE+-]+)"},
    {"Name": "val:recall", "Regex": r"Recall: ([0-9.eE+-]+)"},
    {"Name": "val:f1", "Regex": r"F1: ([0-9.eE+-]+)"},
] + list(cloudwatch_health_metric_definitions())

# Default container images (AWS-managed PyTorch DLC).
# Update the patch version when AWS releases new ones.
PYTORCH_DLC = {
    "us-east-1": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.1.0-gpu-py310-cu121-ubuntu20.04-sagemaker",
    "us-west-2": "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.1.0-gpu-py310-cu121-ubuntu20.04-sagemaker",
    "eu-west-1": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/pytorch-training:2.1.0-gpu-py310-cu121-ubuntu20.04-sagemaker",
}
PYTORCH_INFERENCE_DLC = {
    "us-east-1": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.1.0-gpu-py310-cu121-ubuntu20.04-sagemaker",
    "us-west-2": "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:2.1.0-gpu-py310-cu121-ubuntu20.04-sagemaker",
}


def _csv_env_ids(name: str) -> tuple[str, ...]:
    """Parse comma-separated subnet or security group IDs from an env var."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return ()
    return tuple(x.strip() for x in raw.split(",") if x.strip())


@dataclass
class SMConfig:
    """All SageMaker configuration in one place."""

    bucket: str
    prefix: str = "maxsight"
    region: str = "us-east-1"
    role_arn: str = ""
    instance_type_train: str = "ml.g5.2xlarge"
    instance_type_infer: str = "ml.g5.xlarge"
    instance_count: int = 1
    volume_size_gb: int = 100
    max_run_hours: int = 48
    output_s3_prefix: str = ""
    tags: list[dict[str, str]] = field(default_factory=list)
    # VPC-only training/inference: set both SM_SUBNET_IDS and SM_SECURITY_GROUP_IDS (comma-separated).
    subnets: tuple[str, ...] = ()
    security_group_ids: tuple[str, ...] = ()
    # Optional KMS key ARNs for volume and endpoint model artifacts (org policy).
    volume_kms_key_id: str = ""

    @classmethod
    def from_env(cls) -> SMConfig:
        """Load from environment variables (with sensible defaults)."""
        return cls(
            bucket=os.environ.get("MAXSIGHT_S3_BUCKET", "maxsight-ml"),
            prefix=os.environ.get("MAXSIGHT_S3_PREFIX", "maxsight"),
            region=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
            role_arn=os.environ.get("SAGEMAKER_ROLE_ARN", ""),
            instance_type_train=os.environ.get("SM_TRAIN_INSTANCE", "ml.g5.2xlarge"),
            instance_type_infer=os.environ.get("SM_INFER_INSTANCE", "ml.g5.xlarge"),
            max_run_hours=int(os.environ.get("SM_MAX_HOURS", "48")),
            subnets=_csv_env_ids("SM_SUBNET_IDS"),
            security_group_ids=_csv_env_ids("SM_SECURITY_GROUP_IDS"),
            volume_kms_key_id=os.environ.get("SM_VOLUME_KMS_KEY_ID", "").strip(),
        )

    @property
    def output_path(self) -> str:
        pfx = self.output_s3_prefix or self.prefix
        return f"s3://{self.bucket}/{pfx}/output"

    @property
    def checkpoint_s3_path(self) -> str:
        return f"s3://{self.bucket}/{self.prefix}/checkpoints"

    @property
    def training_image(self) -> str:
        return PYTORCH_DLC.get(self.region, PYTORCH_DLC["us-east-1"])

    @property
    def inference_image(self) -> str:
        return PYTORCH_INFERENCE_DLC.get(self.region, PYTORCH_INFERENCE_DLC["us-east-1"])


# ── Session / role ────────────────────────────────────────────────────────────


def get_session(region: str | None = None):
    """Return a SageMaker Session (creates boto3 Session internally)."""
    import boto3
    import sagemaker  # type: ignore

    boto_session = boto3.Session(region_name=region)
    return sagemaker.Session(boto_session=boto_session)


def _assert_role_matches_caller(role_arn: str) -> None:
    """Raise RuntimeError if the role ARN account differs from the caller's AWS account.

    Catches wrong-account deploys before anything is submitted. Skip by setting
    MAXSIGHT_SKIP_ROLE_ASSERT=1 (e.g. intentional cross-account scenarios).
    See infra/iam/sagemaker_execution_role.json for the expected trust policy.
    """
    if os.environ.get("MAXSIGHT_SKIP_ROLE_ASSERT", "").strip() == "1":
        return
    match = re.match(r"arn:aws(?:-[a-z]+)*:iam::(\d+):", role_arn)
    if not match:
        # Non-standard ARN (e.g. GovCloud partition); skip rather than false-positive.
        return
    role_account = match.group(1)
    try:
        import boto3

        caller = boto3.client("sts").get_caller_identity()
        caller_account = caller["Account"]
    except Exception:
        # No credentials available (local dev without AWS config); skip assertion.
        return
    if role_account != caller_account:
        raise RuntimeError(
            f"Role ARN account {role_account} does not match caller account {caller_account}. "
            "Check SAGEMAKER_ROLE_ARN. See infra/iam/sagemaker_execution_role.json. "
            "Set MAXSIGHT_SKIP_ROLE_ASSERT=1 to disable this check for cross-account use."
        )


def get_execution_role(role_arn: str = "") -> str:
    """Return a SageMaker execution role ARN.

    Falls back to the attached IAM role when running inside SageMaker.
    Validates that the resolved ARN belongs to the caller's AWS account unless
    MAXSIGHT_SKIP_ROLE_ASSERT=1. See infra/iam/sagemaker_execution_role.json.
    """
    resolved: str
    if role_arn:
        resolved = role_arn
    else:
        role_env = os.environ.get("SAGEMAKER_ROLE_ARN", "")
        if role_env:
            resolved = role_env
        else:
            try:
                import sagemaker  # type: ignore

                resolved = sagemaker.get_execution_role()
            except Exception:
                raise RuntimeError(
                    "Cannot determine SageMaker execution role. "
                    "Set SAGEMAKER_ROLE_ARN env var or run inside a SageMaker context."
                )
    _assert_role_matches_caller(resolved)
    return resolved


# ── Training job config builder ───────────────────────────────────────────────


def _default_source_dir() -> str:
    # ml/infra/sagemaker_utils.py -> repo root
    return str(Path(__file__).resolve().parents[2])


def _optional_debugger_hook(enable: bool):
    if not enable:
        return None
    try:
        from sagemaker.debugger import CollectionConfig, DebuggerHookConfig  # type: ignore

        return DebuggerHookConfig(
            collection_configs=[
                CollectionConfig(name="losses", parameters={"save_interval": "10"}),
                CollectionConfig(name="weights", parameters={"save_interval": "500"}),
            ]
        )
    except Exception as e:
        logger.warning("SageMaker Debugger not configured (%s); continuing without debugger.", e)
        return None


def build_estimator(
    cfg: SMConfig,
    entry_point: str = "ml/training/sagemaker_entry.py",
    hyperparameters: dict[str, Any] | None = None,
    *,
    source_dir: str | None = None,
    dependencies: list[str] | None = None,
    use_spot: bool = False,
    metric_definitions: list[dict[str, str]] | None = None,
    emit_cloudwatch_metrics: bool = True,
    enable_debugger: bool = False,
):
    """Build a SageMaker PyTorch Estimator from SMConfig.

    Pass ``source_dir`` as the repository root so ``entry_point`` (e.g. ``ml/training/sagemaker_entry.py``)
    resolves inside the uploaded tarball. Defaults to this repo root.
    """
    from sagemaker.pytorch import PyTorch  # type: ignore

    session = get_session(cfg.region)
    role = get_execution_role(cfg.role_arn)

    spot_kwargs: dict[str, Any] = {}
    if use_spot:
        spot_kwargs = {
            "use_spot_instances": True,
            "max_wait": cfg.max_run_hours * 3600,
        }

    dbg = _optional_debugger_hook(enable_debugger)
    extra_estimator_kwargs: dict[str, Any] = {}
    if dbg is not None:
        extra_estimator_kwargs["debugger_hook_config"] = dbg

    if metric_definitions is not None:
        metrics: list[dict[str, str]] | None = metric_definitions
    elif emit_cloudwatch_metrics:
        metrics = TRAINING_METRIC_DEFINITIONS
    else:
        metrics = None

    pytorch_kwargs: dict[str, Any] = dict(
        entry_point=entry_point,
        source_dir=source_dir or _default_source_dir(),
        role=role,
        instance_type=cfg.instance_type_train,
        instance_count=cfg.instance_count,
        volume_size=cfg.volume_size_gb,
        framework_version="2.1",
        py_version="py310",
        output_path=cfg.output_path,
        checkpoint_s3_uri=cfg.checkpoint_s3_path,
        hyperparameters=hyperparameters or {},
        sagemaker_session=session,
        max_run=cfg.max_run_hours * 3600,
        tags=cfg.tags,
        dependencies=dependencies or [],
        **spot_kwargs,
        **extra_estimator_kwargs,
    )
    if metrics:
        pytorch_kwargs["metric_definitions"] = metrics

    if cfg.subnets and cfg.security_group_ids:
        pytorch_kwargs["subnets"] = list(cfg.subnets)
        pytorch_kwargs["security_group_ids"] = list(cfg.security_group_ids)
    elif cfg.subnets or cfg.security_group_ids:
        logger.warning(
            "SM_SUBNET_IDS and SM_SECURITY_GROUP_IDS must both be set for VPC training; ignoring partial VPC config."
        )

    if cfg.volume_kms_key_id:
        pytorch_kwargs["volume_kms_key"] = cfg.volume_kms_key_id

    estimator = PyTorch(**pytorch_kwargs)
    return estimator


def build_data_channels(
    cfg: SMConfig,
    *,
    train_s3_uri: str | None = None,
    val_s3_uri: str | None = None,
    gold_index_s3_uri: str | None = None,
) -> dict[str, Any]:
    """Build SageMaker data channel inputs for a training job."""
    from sagemaker.inputs import TrainingInput  # type: ignore

    channels: dict[str, Any] = {}
    if train_s3_uri:
        channels["train"] = TrainingInput(train_s3_uri, content_type="application/json")
    if val_s3_uri:
        channels["val"] = TrainingInput(val_s3_uri, content_type="application/json")
    if gold_index_s3_uri:
        channels["gold"] = TrainingInput(gold_index_s3_uri, content_type="application/json")
    return channels


# ── Endpoint helpers ──────────────────────────────────────────────────────────


def deploy_model(
    cfg: SMConfig,
    model_data_s3: str,
    endpoint_name: str = "maxsight-inference",
    *,
    model_server_workers: int = 2,
):
    """Deploy a trained model to a real-time SageMaker endpoint."""
    from sagemaker.pytorch import PyTorchModel  # type: ignore

    session = get_session(cfg.region)
    role = get_execution_role(cfg.role_arn)

    model = PyTorchModel(
        model_data=model_data_s3,
        role=role,
        framework_version="2.1",
        py_version="py310",
        entry_point="ml/infra/inference_handler.py",
        sagemaker_session=session,
        model_server_workers=model_server_workers,
    )
    deploy_kwargs: dict[str, Any] = dict(
        initial_instance_count=1,
        instance_type=cfg.instance_type_infer,
        endpoint_name=endpoint_name,
    )
    if cfg.subnets and cfg.security_group_ids:
        deploy_kwargs["vpc_config_override"] = {
            "Subnets": list(cfg.subnets),
            "SecurityGroupIds": list(cfg.security_group_ids),
        }
    elif cfg.subnets or cfg.security_group_ids:
        logger.warning(
            "SM_SUBNET_IDS and SM_SECURITY_GROUP_IDS must both be set for VPC inference; deploying without vpc_config_override."
        )

    predictor = model.deploy(**deploy_kwargs)
    logger.info("Endpoint live: %s", endpoint_name)
    return predictor


def get_latest_training_job_output(job_name: str, cfg: SMConfig) -> str:
    """Return the S3 URI for the model.tar.gz from a completed training job."""
    import boto3

    sm = boto3.client("sagemaker", region_name=cfg.region)
    resp = sm.describe_training_job(TrainingJobName=job_name)
    return resp["ModelArtifacts"]["S3ModelArtifacts"]


# ── Spot-instance resume helper ───────────────────────────────────────────────


def checkpoint_s3_uri_for_run(cfg: SMConfig, run_id: str) -> str:
    return f"{cfg.checkpoint_s3_path}/{run_id}"


def latest_checkpoint_key(cfg: SMConfig, run_id: str) -> str | None:
    """Find the most recent checkpoint key for a run in S3."""
    from ml.infra.s3_client import S3Client

    client = S3Client(cfg.bucket, cfg.prefix)
    keys = client.list_checkpoints(run_id)
    if not keys:
        return None
    return sorted(keys)[-1]
