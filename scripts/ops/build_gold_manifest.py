#!/usr/bin/env python3
"""Build versioned gold JSONL manifests from raw annotations (training data plane)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ml.data.gold.builder import build_gold_manifest, write_manifest_meta  # noqa: E402
from ml.data.gold.dataNormalizationLayer import COCOAdapter, MaxSightListAdapter  # noqa: E402
from ml.data.gold.label_mapper import LabelMapper  # noqa: E402


def _cmd_maxsight_list(ns: argparse.Namespace) -> int:
    adapter = MaxSightListAdapter(
        Path(ns.annotation),
        Path(ns.image_root),
        Path(ns.repo_root),
        dataset_id=ns.dataset_id,
        version=ns.version,
        split=ns.split,
    )
    mapper = LabelMapper(ns.source_label_space, ns.label_space)
    out = Path(ns.out)
    if ns.num_shards > 1:
        out.mkdir(parents=True, exist_ok=True)
    summary = build_gold_manifest(
        adapter,
        mapper=mapper,
        out=out,
        repo_root=Path(ns.repo_root),
        source_annotation=str(Path(ns.annotation).resolve()),
        num_shards=ns.num_shards,
        skip_invalid=not ns.fail_on_invalid,
    )
    meta_path = (
        Path(ns.meta_out)
        if ns.meta_out
        else (
            out.parent / f"{out.name}_meta.json"
            if ns.num_shards > 1
            else out.parent / f"{out.stem}_meta.json"
        )
    )
    write_manifest_meta(
        meta_path,
        repo_root=Path(ns.repo_root),
        dataset_id=ns.dataset_id,
        version=ns.version,
        split=ns.split,
        label_space=ns.label_space,
        num_classes=ns.num_classes,
        class_map_hash=summary["class_map_hash"],
        source_annotation=str(Path(ns.annotation).resolve()),
        lines_written=summary["lines_written"],
        lines_skipped=summary["lines_skipped"],
        shards=summary["shards"],
    )
    print(
        f"wrote {summary['lines_written']} lines, skipped {summary['lines_skipped']}, "
        f"shards={len(summary['shards'])}, meta={meta_path}"
    )
    return 0


def _cmd_coco_instances(ns: argparse.Namespace) -> int:
    adapter = COCOAdapter(
        Path(ns.annotation),
        Path(ns.image_root),
        Path(ns.repo_root),
        dataset_id=ns.dataset_id,
        version=ns.version,
        split=ns.split,
    )
    mapper = LabelMapper(ns.source_label_space, ns.label_space)
    out = Path(ns.out)
    if ns.num_shards > 1:
        out.mkdir(parents=True, exist_ok=True)
    summary = build_gold_manifest(
        adapter,
        mapper=mapper,
        out=out,
        repo_root=Path(ns.repo_root),
        source_annotation=str(Path(ns.annotation).resolve()),
        num_shards=ns.num_shards,
        skip_invalid=not ns.fail_on_invalid,
    )
    meta_path = (
        Path(ns.meta_out)
        if ns.meta_out
        else (
            out.parent / f"{out.name}_meta.json"
            if ns.num_shards > 1
            else out.parent / f"{out.stem}_meta.json"
        )
    )
    write_manifest_meta(
        meta_path,
        repo_root=Path(ns.repo_root),
        dataset_id=ns.dataset_id,
        version=ns.version,
        split=ns.split,
        label_space=ns.label_space,
        num_classes=ns.num_classes,
        class_map_hash=summary["class_map_hash"],
        source_annotation=str(Path(ns.annotation).resolve()),
        lines_written=summary["lines_written"],
        lines_skipped=summary["lines_skipped"],
        shards=summary["shards"],
    )
    print(
        f"wrote {summary['lines_written']} lines, skipped {summary['lines_skipped']}, "
        f"shards={len(summary['shards'])}, meta={meta_path}"
    )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--repo-root",
        type=Path,
        default=REPO,
        help="Repository root (default: inferred from this script)",
    )
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(c: argparse.ArgumentParser) -> None:
        c.add_argument("--annotation", type=Path, required=True)
        c.add_argument("--image-root", type=Path, required=True)
        c.add_argument("--out", type=Path, required=True)
        c.add_argument("--dataset-id", type=str, required=True)
        c.add_argument("--version", type=str, required=True)
        c.add_argument("--split", type=str, required=True)
        c.add_argument("--label-space", type=str, default="accessibility_622")
        c.add_argument(
            "--num-classes",
            type=int,
            default=622,
            help="Number of classes in the target label space (default: 622 for accessibility_622)",
        )
        c.add_argument(
            "--source-label-space",
            type=str,
            default=None,
            help="Reserved for future remaps; omit for name→622 mapping.",
        )
        c.add_argument(
            "--num-shards",
            type=int,
            default=1,
            help=">1 writes shard_00000.jsonl under --out directory",
        )
        c.add_argument("--meta-out", type=Path, default=None)
        c.add_argument(
            "--fail-on-invalid",
            action="store_true",
            help="Abort on first invalid row instead of skip+log",
        )

    m = sub.add_parser("maxsight-list", help="From MaxSight list JSON to gold JSONL")
    add_common(m)
    m.set_defaults(func=_cmd_maxsight_list)

    c = sub.add_parser(
        "coco-instances",
        help="From COCO instances JSON (images+annotations) to gold JSONL",
    )
    add_common(c)
    c.set_defaults(func=_cmd_coco_instances)

    ns = p.parse_args()
    if ns.num_shards < 1:
        print("--num-shards must be >= 1", file=sys.stderr)
        return 2
    if ns.num_shards > 1 and ns.out.suffix.lower() == ".jsonl":
        print(
            "When --num-shards > 1, --out must be a directory path (not a .jsonl file).",
            file=sys.stderr,
        )
        return 2
    return int(ns.func(ns))


if __name__ == "__main__":
    sys.exit(main())
