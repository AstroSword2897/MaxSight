#!/usr/bin/env python3
"""Submit an *offline* SageMaker Processing job for ``ml/pipeline/sagemaker_entrypoint.py``.

This script is the ops-layer launcher for the offline preprocessing pipeline only.
It must NOT be used to trigger inference or training jobs — use sagemaker_deploy.py
and sagemaker_train.py respectively for those paths.

The input channel must contain ``video_records.json`` (see ``run_sagemaker_pipeline``).
Output is ``phase3_pipeline_output.json`` under the processing output channel.

Usage
-----
python scripts/ops/sagemaker_processing_submit.py \\
  --bucket my-bucket --role arn:aws:iam::123:role/SageMakerRole \\
  --input-s3 s3://my-bucket/prefix/processing-in/ \\
  --output-s3 s3://my-bucket/prefix/processing-out/

python scripts/ops/sagemaker_processing_submit.py ... --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ml.infra.sagemaker_utils import SMConfig, get_execution_role, get_session  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bucket", default="", help="Or MAXSIGHT_S3_BUCKET")
    p.add_argument("--prefix", default="maxsight")
    p.add_argument("--region", default="us-east-1")
    p.add_argument("--role", default="", help="Or SAGEMAKER_ROLE_ARN")
    p.add_argument("--input-s3", required=True, help="S3 URI prefix for processing input (video_records.json + assets)")
    p.add_argument("--output-s3", required=True, help="S3 URI prefix for processing output artefacts")
    p.add_argument("--instance", default="ml.m5.2xlarge", help="CPU or GPU instance for preprocessing")
    p.add_argument("--job-name", default="", help="Override auto job name")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    import os
    import time

    args = parse_args()
    bucket = args.bucket or os.environ.get("MAXSIGHT_S3_BUCKET", "")
    if not bucket:
        print("Error: --bucket or MAXSIGHT_S3_BUCKET required", file=sys.stderr)
        return 1
    role = args.role or os.environ.get("SAGEMAKER_ROLE_ARN", "")
    if not role and not args.dry_run:
        print("Error: --role or SAGEMAKER_ROLE_ARN required", file=sys.stderr)
        return 1

    cfg = SMConfig(bucket=bucket, prefix=args.prefix, region=args.region, role_arn=role)
    job_name = args.job_name or f"maxsight-pipeline-{time.strftime('%Y%m%d-%H%M%S')}"

    payload = {
        "job_name": job_name,
        "framework": "PyTorchProcessor",
        "entry": "ml/pipeline/sagemaker_entrypoint.py",
        "source_dir": str(REPO),
        "input_s3": args.input_s3,
        "output_s3": args.output_s3,
        "instance": args.instance,
    }

    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return 0

    try:
        from sagemaker.processing import ProcessingInput, ProcessingOutput  # type: ignore
        from sagemaker.pytorch.processing import PyTorchProcessor  # type: ignore
    except ImportError as e:
        print("Error: install sagemaker SDK: pip install sagemaker", file=sys.stderr)
        print(e, file=sys.stderr)
        return 1

    session = get_session(cfg.region)
    pytorch_processor = PyTorchProcessor(
        framework_version="2.1.0",
        py_version="py310",
        role=get_execution_role(role),
        instance_type=args.instance,
        instance_count=1,
        base_job_name="maxsight-pipeline",
        sagemaker_session=session,
    )

    pytorch_processor.run(
        code="ml/pipeline/sagemaker_entrypoint.py",
        source_dir=str(REPO),
        inputs=[
            ProcessingInput(
                input_name="train",
                source=args.input_s3.rstrip("/"),
                destination="/opt/ml/processing/input/train",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="train",
                source="/opt/ml/processing/output/train",
                destination=args.output_s3.rstrip("/"),
            ),
        ],
        job_name=job_name,
        wait=False,
    )
    print(f"Submitted processing job: {job_name}")
    print(
        f"Console: https://{args.region}.console.aws.amazon.com/sagemaker/home?region={args.region}#/processing-jobs/{job_name}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
