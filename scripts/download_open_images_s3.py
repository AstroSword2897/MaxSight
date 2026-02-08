#!/usr/bin/env python3
"""Download Open Images from S3 bucket s3://open-images-dataset.

Requires AWS CLI installed and configured (aws configure).
Syncs to datasets/open_images_v6 by default so inference scripts find the data.

Usage:
  python scripts/download_open_images_s3.py
  python scripts/download_open_images_s3.py --prefix validation/
  python scripts/download_open_images_s3.py --dest datasets/open_images_v6 --dry-run
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUCKET = "s3://open-images-dataset"
DEFAULT_DEST = ROOT / "datasets" / "open_images_v6"


def main():
    parser = argparse.ArgumentParser(
        description="Download Open Images from s3://open-images-dataset via AWS CLI."
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"Local directory to sync into (default: {DEFAULT_DEST})",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="S3 prefix to sync (e.g. validation/ to download only validation split)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be downloaded (aws s3 sync --dryrun)",
    )
    args = parser.parse_args()

    dest = args.dest.resolve()
    if not shutil.which("aws"):
        print("AWS CLI is required. Install: https://aws.amazon.com/cli/")
        print("Then run: aws configure")
        sys.exit(1)

    s3_uri = BUCKET if not args.prefix else f"{BUCKET}/{args.prefix.rstrip('/')}"
    dest.mkdir(parents=True, exist_ok=True)

    cmd = ["aws", "s3", "sync", s3_uri, str(dest), "--no-sign-request"]
    if args.dry_run:
        cmd.append("--dryrun")

    print(f"Syncing {s3_uri} -> {dest}")
    if args.dry_run:
        print("(dry run)")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("Sync failed. If the bucket is not public, use: aws s3 sync ... (without --no-sign-request)")
        sys.exit(result.returncode)
    print("Done.")


if __name__ == "__main__":
    main()

