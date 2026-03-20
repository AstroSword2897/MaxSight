"""SageMaker session, role, and job-configuration helpers.

Abstracts the SageMaker SDK so the rest of the codebase can reference
``get_session()`` / ``get_execution_role()`` without touching boto3 directly.

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

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
    tags: List[Dict[str, str]] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> "SMConfig":
        """Load from environment variables (with sensible defaults)."""
        return cls(
            bucket=os.environ.get("MAXSIGHT_S3_BUCKET", "maxsight-ml"),
            prefix=os.environ.get("MAXSIGHT_S3_PREFIX", "maxsight"),
            region=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
            role_arn=os.environ.get("SAGEMAKER_ROLE_ARN", ""),
            instance_type_train=os.environ.get("SM_TRAIN_INSTANCE", "ml.g5.2xlarge"),
            instance_type_infer=os.environ.get("SM_INFER_INSTANCE", "ml.g5.xlarge"),
            max_run_hours=int(os.environ.get("SM_MAX_HOURS", "48")),
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

def get_session(region: Optional[str] = None):
    """Return a SageMaker Session (creates boto3 Session internally)."""
    import boto3
    import sagemaker  # type: ignore

    boto_session = boto3.Session(region_name=region)
    return sagemaker.Session(boto_session=boto_session)


def get_execution_role(role_arn: str = "") -> str:
    """Return a SageMaker execution role ARN.

    Falls back to the attached IAM role when running inside SageMaker.
    """
    if role_arn:
        return role_arn
    role_env = os.environ.get("SAGEMAKER_ROLE_ARN", "")
    if role_env:
        return role_env
    try:
        import sagemaker  # type: ignore
        return sagemaker.get_execution_role()
    except Exception:
        raise RuntimeError(
            "Cannot determine SageMaker execution role. "
            "Set SAGEMAKER_ROLE_ARN env var or run inside a SageMaker context."
        )


# ── Training job config builder ───────────────────────────────────────────────

def build_estimator(
    cfg: SMConfig,
    entry_point: str = "ml/training/sagemaker_entry.py",
    hyperparameters: Optional[Dict[str, Any]] = None,
    *,
    source_dir: Optional[str] = None,
    dependencies: Optional[List[str]] = None,
    use_spot: bool = False,
):
    """Build a SageMaker PyTorch Estimator from SMConfig."""
    import sagemaker  # type: ignore
    from sagemaker.pytorch import PyTorch  # type: ignore

    session = get_session(cfg.region)
    role = get_execution_role(cfg.role_arn)

    spot_kwargs: Dict[str, Any] = {}
    if use_spot:
        spot_kwargs = {
            "use_spot_instances": True,
            "max_wait": cfg.max_run_hours * 3600,
        }

    estimator = PyTorch(
        entry_point=entry_point,
        source_dir=source_dir,
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
    )
    return estimator


def build_data_channels(
    cfg: SMConfig,
    *,
    train_s3_uri: Optional[str] = None,
    val_s3_uri: Optional[str] = None,
    gold_index_s3_uri: Optional[str] = None,
) -> Dict[str, Any]:
    """Build SageMaker data channel inputs for a training job."""
    from sagemaker.inputs import TrainingInput  # type: ignore

    channels: Dict[str, Any] = {}
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
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=cfg.instance_type_infer,
        endpoint_name=endpoint_name,
    )
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


def latest_checkpoint_key(cfg: SMConfig, run_id: str) -> Optional[str]:
    """Find the most recent checkpoint key for a run in S3."""
    from ml.infra.s3_client import S3Client

    client = S3Client(cfg.bucket, cfg.prefix)
    keys = client.list_checkpoints(run_id)
    if not keys:
        return None
    return sorted(keys)[-1]
