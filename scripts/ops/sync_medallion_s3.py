#!/usr/bin/env python3
"""Sync the local medallion data lake (bronze/silver/gold) to/from S3.

All layers are mirrored under:
  s3://<BUCKET>/<PREFIX>/medallion/<layer>/

Usage
-----
# Upload gold index only (fast check-in after training_index.json changes)
python scripts/ops/sync_medallion_s3.py upload gold --bucket my-bucket

# Upload all silver data
python scripts/ops/sync_medallion_s3.py upload silver --bucket my-bucket

# Pull gold from S3 (e.g. on a new machine)
python scripts/ops/sync_medallion_s3.py download gold --bucket my-bucket

# Sync all layers up (dry-run first)
python scripts/ops/sync_medallion_s3.py upload all --bucket my-bucket --dry-run

# Use a config file
python scripts/ops/sync_medallion_s3.py upload silver \
    --bucket my-bucket --prefix maxsight --region us-west-2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ml.infra.s3_client import S3Client, MEDALLION_LAYERS  # noqa: E402
from ml.data.medallion_layout import default_medallion_root  # noqa: E402

LAYER_CHOICES = list(MEDALLION_LAYERS) + ["all"]


def _progress(msg: str) -> None:
    print(f"  {msg}")


def cmd_upload(args: argparse.Namespace) -> int:
    layers = list(MEDALLION_LAYERS) if args.layer == "all" else [args.layer]
    client = _make_client(args)
    mroot = args.medallion_root.resolve()
    results = []
    for layer in layers:
        local = mroot / layer
        if not local.exists():
            print(f"[skip] {layer}: local dir missing ({local})")
            continue
        if args.dry_run:
            count = sum(1 for p in local.rglob("*") if p.is_file())
            print(f"[dry-run] Would upload {count} files from {local} → medallion/{layer}/")
            continue
        result = client.upload_medallion_layer(layer, mroot, overwrite=args.overwrite)
        results.append(result)
        print(json.dumps(result, indent=2))
    if any(r.get("files_failed", 0) > 0 for r in results):
        return 1
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    layers = list(MEDALLION_LAYERS) if args.layer == "all" else [args.layer]
    client = _make_client(args)
    mroot = args.medallion_root.resolve()
    for layer in layers:
        if args.dry_run:
            prefix = client.medallion_s3_prefix(layer)
            keys = client.list_keys(prefix, max_keys=args.max_list_keys)
            cap_note = f" (capped at {args.max_list_keys})" if args.max_list_keys else ""
            print(f"[dry-run] Would download {len(keys)} files from {prefix}/{cap_note}")
            continue
        result = client.download_medallion_layer(layer, mroot, overwrite=args.overwrite)
        print(json.dumps(result, indent=2))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Show how many objects exist per layer in S3."""
    client = _make_client(args)
    rows = []
    for layer in MEDALLION_LAYERS:
        prefix = client.medallion_s3_prefix(layer)
        keys = client.list_keys(prefix, max_keys=args.max_list_keys)
        row = {
            "layer": layer,
            "s3_objects": len(keys),
            "s3_prefix": f"s3://{args.bucket}/{prefix}",
        }
        if args.max_list_keys and len(keys) >= args.max_list_keys:
            row["truncated"] = True
        rows.append(row)
    print(json.dumps(rows, indent=2))
    return 0


def _make_client(args: argparse.Namespace) -> S3Client:
    return S3Client(
        bucket=args.bucket,
        prefix=args.prefix,
        region=args.region or None,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--prefix", default="maxsight", help="S3 key prefix")
    parser.add_argument("--region", default="", help="AWS region")
    parser.add_argument("--medallion-root", type=Path, default=default_medallion_root(REPO))
    parser.add_argument(
        "--max-list-keys",
        type=int,
        default=None,
        metavar="N",
        help="Cap list_objects per prefix (status/dry-run) to avoid huge memory use on massive buckets",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_up = sub.add_parser("upload", help="Upload local medallion layer(s) to S3")
    p_up.add_argument("layer", choices=LAYER_CHOICES)
    p_up.add_argument("--overwrite", action="store_true", help="Overwrite even if unchanged")
    p_up.add_argument("--dry-run", action="store_true")
    p_up.set_defaults(func=cmd_upload)

    p_dl = sub.add_parser("download", help="Download S3 medallion layer(s) to local")
    p_dl.add_argument("layer", choices=LAYER_CHOICES)
    p_dl.add_argument("--overwrite", action="store_true")
    p_dl.add_argument("--dry-run", action="store_true")
    p_dl.set_defaults(func=cmd_download)

    p_st = sub.add_parser("status", help="Show S3 object count per medallion layer")
    p_st.set_defaults(func=cmd_status)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
