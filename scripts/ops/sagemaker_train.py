#!/usr/bin/env python3
"""Launch a MaxSight training job on SageMaker.

Reads the gold training_index.json, uploads it (and optionally silver data)
to S3, then submits a SageMaker PyTorch training job.

Usage
-----
# Minimal — uses MAXSIGHT_S3_BUCKET and SAGEMAKER_ROLE_ARN env vars
python scripts/ops/sagemaker_train.py \
    --bucket my-maxsight-bucket --role arn:aws:iam::123:role/SageMakerRole

# Full example with a T5 temporal model
python scripts/ops/sagemaker_train.py \
    --bucket my-maxsight-bucket \
    --role arn:aws:iam::123:role/SageMakerRole \
    --tier T5_TEMPORAL \
    --epochs 50 \
    --freeze-backbone \
    --use-spot \
    --instance ml.g5.2xlarge

# Dry run (print job config without submitting)
python scripts/ops/sagemaker_train.py --bucket my-bucket --role ... --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ml.infra.sagemaker_utils import SMConfig, build_estimator, build_data_channels  # noqa: E402
from ml.infra.s3_client import S3Client  # noqa: E402
from ml.data.medallion_layout import default_gold_index_path, default_medallion_root  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # AWS / infra
    parser.add_argument("--bucket", default="", help="S3 bucket (or set MAXSIGHT_S3_BUCKET)")
    parser.add_argument("--prefix", default="maxsight")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--role", default="", help="SageMaker execution role ARN")
    parser.add_argument("--instance", default="ml.g5.2xlarge")
    parser.add_argument("--use-spot", action="store_true", help="Use spot instances (cheaper)")

    # Data
    parser.add_argument("--medallion-root", type=Path, default=default_medallion_root(REPO))
    parser.add_argument("--upload-silver", action="store_true",
                        help="Upload silver data to S3 before submitting the job")

    # Tier YAML config drives all training-affecting fields; --epochs etc.
    # are explicit overrides only and feed through SM_HP_* on the container.
    parser.add_argument("--config", required=True,
                        help="Path to tier YAML (e.g. ml/training/configs/t5_temporal.yaml); "
                             "the file is shipped via source_dir and read inside the container")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training.num_epochs (otherwise YAML wins)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override data.batch_size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override training.learning_rate")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Override training.freeze_backbone=true")
    parser.add_argument("--freeze-backbone-epochs", type=int, default=None,
                        help="Override training.freeze_backbone_epochs")
    parser.add_argument("--fp16", action="store_true",
                        help="Override training.mixed_precision=true")

    parser.add_argument("--experiment", default="maxsight")
    parser.add_argument("--job-name", default="", help="Override auto-generated job name")
    parser.add_argument("--dry-run", action="store_true", help="Print config; do not submit")
    parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="Disable CloudWatch metric_definitions (regex on training logs)",
    )
    parser.add_argument(
        "--debugger",
        action="store_true",
        help="Enable SageMaker Debugger hook (loss/weights collections) if SDK supports it",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import os
    bucket = args.bucket or os.environ.get("MAXSIGHT_S3_BUCKET", "")
    if not bucket:
        print("Error: --bucket required or set MAXSIGHT_S3_BUCKET", file=sys.stderr)
        return 1

    role = args.role or os.environ.get("SAGEMAKER_ROLE_ARN", "")

    cfg = SMConfig(
        bucket=bucket,
        prefix=args.prefix,
        region=args.region,
        role_arn=role,
        instance_type_train=args.instance,
    )

    s3 = S3Client(bucket=bucket, prefix=args.prefix, region=args.region)
    mroot = args.medallion_root.resolve()
    gold_index_path = default_gold_index_path(mroot)

    # Upload gold index.
    if gold_index_path.exists():
        gold_s3 = s3.upload_gold_index(gold_index_path)
        print(f"Gold index uploaded: {gold_s3}")
    else:
        print(f"Warning: gold index not found at {gold_index_path}", file=sys.stderr)
        gold_s3 = ""

    # Optionally upload silver.
    if args.upload_silver:
        result = s3.upload_medallion_layer("silver", mroot)
        print(f"Silver uploaded: {result['files_uploaded']} files")

    # SageMaker passes both --<arg> CLI flags and SM_HP_<ARG> env vars to
    # the entrypoint; we only forward fields the operator explicitly set so
    # the YAML referenced by --config remains the source of truth.
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"Error: --config not found: {config_path}", file=sys.stderr)
        return 1
    try:
        rel_config = config_path.relative_to(REPO)
    except ValueError:
        print(
            f"Error: --config {config_path} must live inside the repo "
            "({REPO}) so it ships in source_dir.",
            file=sys.stderr,
        )
        return 1

    hyperparameters: dict[str, str] = {
        "config": str(rel_config),
        "experiment": args.experiment,
    }
    if args.epochs is not None:
        hyperparameters["epochs"] = str(args.epochs)
    if args.batch_size is not None:
        hyperparameters["batch-size"] = str(args.batch_size)
    if args.lr is not None:
        hyperparameters["lr"] = str(args.lr)
    if args.freeze_backbone:
        hyperparameters["freeze-backbone"] = "true"
    if args.freeze_backbone_epochs is not None:
        hyperparameters["freeze-backbone-epochs"] = str(args.freeze_backbone_epochs)
    if args.fp16:
        hyperparameters["fp16"] = "true"

    tier_slug = config_path.stem.replace("_", "-")
    job_name = args.job_name or f"maxsight-{tier_slug}-{time.strftime('%Y%m%d-%H%M%S')}"

    job_config = {
        "job_name": job_name,
        "bucket": bucket,
        "instance": args.instance,
        "use_spot": args.use_spot,
        "hyperparameters": hyperparameters,
        "gold_s3": gold_s3,
        "output_path": cfg.output_path,
    }

    if args.dry_run:
        print("\n[dry-run] Would submit SageMaker training job:")
        job_config["source_dir"] = str(REPO)
        job_config["metric_definitions"] = "default" if not args.no_metrics else "disabled"
        job_config["debugger"] = bool(args.debugger)
        print(json.dumps(job_config, indent=2))
        return 0

    if not role:
        print("Error: --role or SAGEMAKER_ROLE_ARN required for actual job submission", file=sys.stderr)
        return 1

    # Build data channels.
    channels = build_data_channels(cfg, gold_index_s3_uri=gold_s3) if gold_s3 else {}

    # Build and submit estimator.
    estimator = build_estimator(
        cfg,
        entry_point="ml/training/sagemaker_entry.py",
        hyperparameters=hyperparameters,
        use_spot=args.use_spot,
        source_dir=str(REPO),
        emit_cloudwatch_metrics=not args.no_metrics,
        enable_debugger=args.debugger,
    )

    print(f"\nSubmitting training job: {job_name}")
    estimator.fit(channels or None, job_name=job_name, wait=False, logs="None")
    print(f"Job submitted: {job_name}")
    print(f"Monitor at: https://{args.region}.console.aws.amazon.com/sagemaker/home#/jobs/{job_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
