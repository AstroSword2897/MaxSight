#!/usr/bin/env python3
"""Create a SageMaker Model Package Group (one-time per environment).

Use the same name as MAXSIGHT_MODEL_PACKAGE_GROUP so register_model can submit packages.

Usage
-----
export AWS_DEFAULT_REGION=us-east-1
python scripts/ops/create_model_package_group.py --name maxsight-models-prod --dry-run
python scripts/ops/create_model_package_group.py --name maxsight-models-prod
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--name", required=True, help="ModelPackageGroupName (e.g. maxsight-models-prod)"
    )
    p.add_argument("--region", default="", help="Or AWS_DEFAULT_REGION")
    p.add_argument("--description", default="MaxSight model packages (manual approval)")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    import os

    args = parse_args()
    region = args.region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    payload = {
        "action": "create_model_package_group",
        "ModelPackageGroupName": args.name,
        "ModelPackageGroupDescription": args.description,
        "region": region,
    }
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return 0

    try:
        import boto3
    except ImportError:
        print("Error: pip install boto3", file=sys.stderr)
        return 1

    client = boto3.client("sagemaker", region_name=region)
    try:
        client.create_model_package_group(
            ModelPackageGroupName=args.name,
            ModelPackageGroupDescription=args.description,
        )
        print(f"Created model package group: {args.name}")
    except client.exceptions.ResourceInUse:  # type: ignore[attr-defined]
        print(f"Model package group already exists: {args.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
