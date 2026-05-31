#!/usr/bin/env python3
"""Deploy a trained MaxSight model to a SageMaker real-time endpoint or run batch transform.

Usage
-----
# Deploy best checkpoint from a completed training job
python scripts/ops/sagemaker_deploy.py deploy \
    --bucket my-bucket --role arn:aws:iam::123:role/SageMakerRole \
    --job-name maxsight-t5-temporal-20260301-120000

# Deploy a specific model.tar.gz
python scripts/ops/sagemaker_deploy.py deploy \
    --bucket my-bucket --role ... \
    --model-data s3://my-bucket/maxsight/output/model.tar.gz

# Invoke the endpoint with a test image
python scripts/ops/sagemaker_deploy.py invoke \
    --endpoint maxsight-inference --image tests/fixtures/test_frame.jpg

# Delete the endpoint when done
python scripts/ops/sagemaker_deploy.py delete --endpoint maxsight-inference

# Batch transform over a folder of images in S3
python scripts/ops/sagemaker_deploy.py batch \
    --bucket my-bucket --role ... \
    --model-data s3://my-bucket/.../model.tar.gz \
    --input s3://my-bucket/frames/ --output s3://my-bucket/predictions/

# List running endpoints
python scripts/ops/sagemaker_deploy.py list
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ml.infra.model_registry import ModelEntry, ModelRegistry  # noqa: E402
from ml.infra.sagemaker_utils import (  # noqa: E402
    SMConfig,
    deploy_model,
    get_execution_role,
    get_latest_training_job_output,
    get_session,
)


def _normalize_s3(uri: str) -> str:
    """Normalise S3 URIs so trailing slashes and case don't cause false negatives."""
    return uri.rstrip("/").lower()


def _lookup_by_s3(reg: ModelRegistry, s3_uri: str) -> ModelEntry | None:
    """Return the registry entry whose checkpoint_path matches s3_uri, or None."""
    target = _normalize_s3(s3_uri)
    for entry in reg.list_models():
        if _normalize_s3(str(entry.checkpoint_path)) == target:
            return entry
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bucket", default="")
    parser.add_argument("--prefix", default="maxsight")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--role", default="")

    sub = parser.add_subparsers(dest="command", required=True)

    # deploy
    p_dep = sub.add_parser("deploy", help="Deploy model to real-time endpoint")
    p_dep.add_argument("--endpoint", default="maxsight-inference")
    p_dep.add_argument("--model-data", default="", help="S3 URI to model.tar.gz")
    p_dep.add_argument("--job-name", default="", help="SageMaker training job to pull model from")
    p_dep.add_argument("--instance", default="ml.g5.xlarge")
    p_dep.add_argument("--workers", type=int, default=2)
    p_dep.add_argument(
        "--register", action="store_true", help="Register deployed model in local registry"
    )
    p_dep.add_argument("--tier", default="", help="Tier label for registry")
    p_dep.add_argument(
        "--skip-registry-check",
        action="store_true",
        help="Bypass the model registry gate. USE FOR EMERGENCIES ONLY — emits a loud warning.",
    )
    p_dep.add_argument(
        "--dry-run", action="store_true", help="Print deploy config without submitting"
    )
    p_dep.set_defaults(func=cmd_deploy)

    # invoke
    p_inv = sub.add_parser("invoke", help="Send a frame to a live endpoint")
    p_inv.add_argument("--endpoint", required=True)
    p_inv.add_argument("--image", type=Path, default=None, help="Local image file to send")
    p_inv.add_argument("--payload", type=str, default="", help="Raw JSON payload string")
    p_inv.set_defaults(func=cmd_invoke)

    # delete
    p_del = sub.add_parser("delete", help="Delete a SageMaker endpoint")
    p_del.add_argument("--endpoint", required=True)
    p_del.set_defaults(func=cmd_delete)

    # batch
    p_bat = sub.add_parser("batch", help="Run a batch transform job")
    p_bat.add_argument("--model-data", required=True)
    p_bat.add_argument("--input", required=True, help="S3 URI to input data")
    p_bat.add_argument("--output", required=True, help="S3 URI for output")
    p_bat.add_argument("--job-name", default="")
    p_bat.add_argument("--instance", default="ml.g5.xlarge")
    p_bat.set_defaults(func=cmd_batch)

    # list
    p_lst = sub.add_parser("list", help="List SageMaker endpoints")
    p_lst.set_defaults(func=cmd_list)

    return parser.parse_args()


def _cfg(args: argparse.Namespace) -> SMConfig:
    import os

    base = SMConfig.from_env()
    bucket = args.bucket or os.environ.get("MAXSIGHT_S3_BUCKET", "")
    role = args.role or os.environ.get("SAGEMAKER_ROLE_ARN", "")
    inst = getattr(args, "instance", None) or base.instance_type_infer
    return dataclasses.replace(
        base,
        bucket=bucket or base.bucket,
        prefix=args.prefix or base.prefix,
        region=args.region or base.region,
        role_arn=role or base.role_arn,
        instance_type_infer=inst,
    )


def cmd_deploy(args: argparse.Namespace) -> int:
    import os

    env_profile = os.environ.get("MAXSIGHT_ENV", "").strip().lower()
    if args.skip_registry_check and env_profile in ("production", "prod"):
        print(
            "ERROR: --skip-registry-check is not allowed when MAXSIGHT_ENV=production (or prod).",
            file=sys.stderr,
        )
        return 1

    cfg = _cfg(args)

    model_data = args.model_data
    if not model_data and args.job_name:
        print(f"Fetching model artefacts from job: {args.job_name}")
        model_data = get_latest_training_job_output(args.job_name, cfg)
        print(f"Model data: {model_data}")

    if not model_data:
        print("Error: --model-data or --job-name required", file=sys.stderr)
        return 1

    # Registry gate: every artifact must be registered before it can be deployed.
    # --skip-registry-check exists only for emergency hotfixes — it always prints a warning.
    if args.skip_registry_check:
        print(
            "WARNING: --skip-registry-check bypasses the registry gate. "
            "Ensure this artifact has been validated before deploying to production.",
            file=sys.stderr,
        )
    else:
        reg = ModelRegistry()
        if _lookup_by_s3(reg, model_data) is None:
            print(
                f"ERROR: artifact not in registry: {model_data}\n"
                "Register it first with --register on a previous deploy, or promote via ModelRegistry.\n"
                "Use --skip-registry-check only for emergency deploys.",
                file=sys.stderr,
            )
            return 1

    if getattr(args, "dry_run", False):
        out = {
            "action": "deploy",
            "model_data": model_data,
            "endpoint": args.endpoint,
            "instance": args.instance,
            "workers": args.workers,
        }
        if cfg.subnets and cfg.security_group_ids:
            out["vpc"] = {
                "subnets": list(cfg.subnets),
                "security_group_ids": list(cfg.security_group_ids),
            }
        print(json.dumps(out, indent=2))
        return 0

    cfg.instance_type_infer = args.instance
    predictor = deploy_model(cfg, model_data, args.endpoint, model_server_workers=args.workers)
    print(f"Endpoint deployed: {args.endpoint}")
    print(
        f"Invoke URL: https://runtime.sagemaker.{cfg.region}.amazonaws.com/endpoints/{args.endpoint}/invocations"
    )

    if args.register:
        registry = ModelRegistry()
        registry.register_model(
            run_id=args.job_name or args.endpoint,
            checkpoint_path=Path(model_data),
            tier=args.tier,
            tags={"endpoint": args.endpoint},
            notes=f"Deployed to {args.endpoint}",
        )
        registry.promote_model(args.job_name or args.endpoint, stage="production")
        print("Model registered and promoted to production.")
    return 0


def cmd_invoke(args: argparse.Namespace) -> int:
    import boto3

    sm_runtime = boto3.client("sagemaker-runtime", region_name=args.region)

    if args.image and Path(args.image).exists():
        with open(args.image, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        payload = json.dumps({"image_b64": img_b64})
    elif args.payload:
        payload = args.payload
    else:
        payload = json.dumps({"ping": True})

    resp = sm_runtime.invoke_endpoint(
        EndpointName=args.endpoint,
        ContentType="application/json",
        Body=payload.encode(),
    )
    result = json.loads(resp["Body"].read())
    print(json.dumps(result, indent=2))
    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    import boto3

    sm = boto3.client("sagemaker", region_name=args.region)
    sm.delete_endpoint(EndpointName=args.endpoint)
    print(f"Endpoint deleted: {args.endpoint}")
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    import time

    cfg = _cfg(args)
    session = get_session(cfg.region)
    role = get_execution_role(cfg.role_arn)

    from sagemaker.pytorch import PyTorchModel  # type: ignore

    model = PyTorchModel(
        model_data=args.model_data,
        role=role,
        framework_version="2.1",
        py_version="py310",
        entry_point="ml/infra/inference_handler.py",
        sagemaker_session=session,
    )
    job_name = args.job_name or f"maxsight-batch-{time.strftime('%Y%m%d-%H%M%S')}"
    transformer = model.transformer(
        instance_count=1,
        instance_type=getattr(args, "instance", "ml.g5.xlarge"),
        output_path=args.output,
    )
    transformer.transform(
        data=args.input,
        content_type="application/json",
        job_name=job_name,
        wait=False,
    )
    print(f"Batch job submitted: {job_name}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    import boto3

    sm = boto3.client("sagemaker", region_name=args.region)
    resp = sm.list_endpoints(MaxResults=20, SortBy="CreationTime", SortOrder="Descending")
    endpoints = resp.get("Endpoints", [])
    print(
        json.dumps(
            [
                {
                    "name": e["EndpointName"],
                    "status": e["EndpointStatus"],
                    "created": str(e["CreationTime"]),
                }
                for e in endpoints
            ],
            indent=2,
        )
    )
    return 0


def main() -> int:
    args = parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
