#!/usr/bin/env python3
"""Create or check an immutable, versioned split manifest (ticket 309).

  python scripts/split_manifest.py create --manifest records.jsonl --version 1 \
      --holdout-environment office_b --reason "initial split" --out datasets/splits/split_v1.json
  python scripts/split_manifest.py check --manifest records.jsonl --split-manifest datasets/splits/split_v1.json

create exits 2 if the version file already exists or a split rule is broken.
check exits 0 only if the split manifest is unedited, covers every group, and nothing leaks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ml.datasets.manifest import read_manifest  # noqa: E402
from ml.datasets.split_manifest import (  # noqa: E402
    SplitManifestError,
    apply_split_manifest,
    build_split_manifest,
    check_leakage,
    read_split_manifest,
    write_split_manifest,
)


def cmd_create(args: argparse.Namespace) -> int:
    records = list(read_manifest(args.manifest))
    try:
        manifest = build_split_manifest(
            records, version=args.version, holdout_environments=tuple(args.holdout_environment),
            reason=args.reason, seed=args.seed,
        )
        write_split_manifest(manifest, args.out)
    except FileExistsError as exc:
        print(f"REFUSED: {exc} (a changed split needs a new version)")
        return 2
    except SplitManifestError as exc:
        print(f"REFUSED: {exc}")
        return 2
    counts: dict[str, int] = {}
    for split in manifest.assignments.values():
        counts[split] = counts.get(split, 0) + 1
    print(f"wrote {args.out}: version {manifest.version}, {len(manifest.assignments)} groups {counts}")
    print(f"content_hash {manifest.content_hash}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    records = list(read_manifest(args.manifest))
    try:
        manifest = read_split_manifest(args.split_manifest)
        split_records = apply_split_manifest(records, manifest)
    except SplitManifestError as exc:
        print(f"LEAKAGE CHECK FAILED: {exc}")
        return 1
    problems = check_leakage(split_records)
    for problem in problems[:25]:
        print("ERROR", problem)
    if problems:
        print("LEAKAGE CHECK FAILED")
        return 1
    print(f"LEAKAGE CHECK PASSED: version {manifest.version}, {len(manifest.assignments)} groups")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    create = sub.add_parser("create", help="build and write a new split manifest version")
    create.add_argument("--manifest", required=True, help="records (manifest JSONL) with group_id and extra.environment")
    create.add_argument("--version", required=True)
    create.add_argument("--holdout-environment", action="append", default=[], help="repeatable; at least one")
    create.add_argument("--reason", required=True)
    create.add_argument("--seed", type=int, default=0)
    create.add_argument("--out", required=True)
    create.set_defaults(func=cmd_create)
    check = sub.add_parser("check", help="verify a split manifest and run the leakage check")
    check.add_argument("--manifest", required=True)
    check.add_argument("--split-manifest", required=True)
    check.set_defaults(func=cmd_check)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
