#!/usr/bin/env python3
"""Run train_maxsight sequentially for each tier YAML using the same gold COCO index."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

DEFAULT_CONFIGS: list[tuple[str, Path]] = [
    ("t0_baseline", REPO / "ml/training/configs/t0_baseline.yaml"),
    ("t1_attention", REPO / "ml/training/configs/t1_attention.yaml"),
    ("t2_hybrid_vit", REPO / "ml/training/configs/t2_hybrid_vit.yaml"),
    ("t3_cross_task", REPO / "ml/training/configs/t3_cross_task.yaml"),
    ("t4_cross_modal", REPO / "ml/training/configs/t4_cross_modal.yaml"),
    ("t5_temporal", REPO / "ml/training/configs/t5_temporal.yaml"),
    ("t5_sec", REPO / "ml/training/configs/t5_sec.yaml"),
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=REPO)
    p.add_argument(
        "--gold-index",
        type=Path,
        default=REPO / "datasets" / "medallion" / "gold" / "training_index.json",
    )
    p.add_argument(
        "--checkpoint-parent", type=Path, default=REPO / "checkpoints" / "medallion_runs"
    )
    p.add_argument(
        "--epochs", type=int, default=2, help="Per-config epochs (raise for real training)"
    )
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--configs",
        type=str,
        default=None,
        help="Comma-separated names from DEFAULT_CONFIGS (default: all)",
    )
    args = p.parse_args()
    rr = args.repo_root.resolve()
    runner = rr / "scripts" / "ops" / "train_from_gold_index.py"
    if not runner.exists():
        print("Missing train_from_gold_index.py", file=sys.stderr)
        return 1

    names = None
    if args.configs:
        names = {x.strip() for x in args.configs.split(",") if x.strip()}
    runs = [(n, c) for n, c in DEFAULT_CONFIGS if names is None or n in names]
    if not runs:
        print("No configs matched.", file=sys.stderr)
        return 1

    args.checkpoint_parent.mkdir(parents=True, exist_ok=True)
    for name, cfg in runs:
        if not cfg.exists():
            print(f"Skip missing config: {cfg}", file=sys.stderr)
            continue
        ckpt_dir = args.checkpoint_parent / name
        cmd = [
            sys.executable,
            str(runner),
            "--repo-root",
            str(rr),
            "--gold-index",
            str(args.gold_index),
            "--config",
            str(cfg),
            "--checkpoint-dir",
            str(ckpt_dir),
            "--epochs",
            str(args.epochs),
            "--device",
            args.device,
            "--batch-size",
            str(args.batch_size),
        ]
        print("===", name, "===", flush=True)
        if args.dry_run:
            print(" ".join(cmd))
            continue
        code = subprocess.call(cmd, cwd=rr)
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    sys.exit(main())
