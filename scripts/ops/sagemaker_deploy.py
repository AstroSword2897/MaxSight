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
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ml.infra.sagemaker_utils import (  # noqa: E402
    SMConfig,
    deploy_model,
    get_latest_training_job_output,
    get_session,
    get_execution_role,
)
from ml.infra.model_registry import ModelRegistry  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    p_dep.add_argument("--register", action="store_true", help="Register deployed model in local registry")
    p_dep.add_argument("--tier", default="", help="Tier label for registry")
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
    bucket = args.bucket or os.environ.get("MAXSIGHT_S3_BUCKET", "")
    role = args.role or os.environ.get("SAGEMAKER_ROLE_ARN", "")
    return SMConfig(
        bucket=bucket,
        prefix=args.prefix,
        region=args.region,
        role_arn=role,
        instance_type_infer=getattr(args, "instance", "ml.g5.xlarge"),
    )


def cmd_deploy(args: argparse.Namespace) -> int:
    cfg = _cfg(args)

    model_data = args.model_data
    if not model_data and args.job_name:
        print(f"Fetching model artefacts from job: {args.job_name}")
        model_data = get_latest_training_job_output(args.job_name, cfg)
        print(f"Model data: {model_data}")

    if not model_data:
        print("Error: --model-data or --job-name required", file=sys.stderr)
        return 1

    cfg.instance_type_infer = args.instance
    predictor = deploy_model(cfg, model_data, args.endpoint, model_server_workers=args.workers)
    print(f"Endpoint deployed: {args.endpoint}")
    print(f"Invoke URL: https://runtime.sagemaker.{cfg.region}.amazonaws.com/endpoints/{args.endpoint}/invocations")

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
    import boto3

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
    print(json.dumps([{"name": e["EndpointName"], "status": e["EndpointStatus"],
                       "created": str(e["CreationTime"])} for e in endpoints], indent=2))
    return 0


def main() -> int:
    args = parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
