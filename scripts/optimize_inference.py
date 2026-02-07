#!/usr/bin/env python3
"""Auto mAP optimizer: sweep confidence and NMS IoU via inference only (no retraining)."""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import itertools

CONF_THRESHOLDS = [0.3, 0.1, 0.05, 0.02, 0.01]
NMS_IOU_VALUES = [0.5, 0.6, 0.7, 0.8]


def parse_map_from_output(output: str) -> float:
    """Extract best mAP@0.5 from log lines like '  cvi: mAP=0.12 mAP@0.5=0.25 ...'."""
    best = -1.0
    for line in output.splitlines():
        # Prefer mAP@0.5 (primary metric)
        m = re.search(r"mAP@0\.5=([\d.]+)", line)
        if m:
            try:
                val = float(m.group(1))
                if val > best:
                    best = val
            except ValueError:
                pass
        else:
            m = re.search(r"\bmAP=([\d.]+)", line)
            if m:
                try:
                    val = float(m.group(1))
                    if val > best:
                        best = val
                except ValueError:
                    pass
    return best


def main():
    try:
        _repo_root = Path(__file__).resolve().parents[1]
    except NameError:
        _repo_root = Path.cwd()
    parser = argparse.ArgumentParser(
        description="Sweep confidence and NMS IoU to find best mAP (inference only, no retraining)."
    )
    parser.add_argument(
        "--val-annotation",
        type=Path,
        default=_repo_root / "datasets" / "cleaned_splits" / "maxsight_val.json",
        help="Validation annotations JSON (default: repo datasets/cleaned_splits/maxsight_val.json)",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=_repo_root / "datasets",
        help="Image root directory (default: repo datasets/)",
    )
    parser.add_argument(
        "--checkpoints-base",
        type=Path,
        default=_repo_root / "checkpoints",
        help="Checkpoints base with checkpoints_<condition> folders (default: repo checkpoints/)",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        nargs="*",
        default=None,
        help="Limit to these conditions (default: all); use one e.g. cvi for faster sweep",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Cap batches per run for quicker sweep (default: all)",
    )
    args = parser.parse_args()

    try:
        repo_root = Path(__file__).resolve().parents[1]
    except NameError:
        repo_root = Path.cwd()
    script = repo_root / "scripts" / "run_checkpoint_inference.py"
    if not script.exists():
        print(f"Error: {script} not found", file=sys.stderr)
        return 1

    best_map = -1.0
    best_config = None
    total = len(CONF_THRESHOLDS) * len(NMS_IOU_VALUES)
    n = 0

    for conf, iou in itertools.product(CONF_THRESHOLDS, NMS_IOU_VALUES):
        n += 1
        print(f"\n[{n}/{total}] conf={conf}, nms_iou={iou}")

        cmd = [
            sys.executable,
            str(script),
            "--val-annotation",
            str(args.val_annotation),
            "--image-dir",
            str(args.image_dir),
            "--checkpoints-base",
            str(args.checkpoints_base),
            "--confidence",
            str(conf),
            "--nms-iou",
            str(iou),
        ]
        if args.conditions:
            cmd += ["--conditions"] + args.conditions
        if args.max_batches is not None:
            cmd += ["--max-batches", str(args.max_batches)]

        result = subprocess.run(
            cmd,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
        )
        output = result.stdout + result.stderr
        if result.returncode != 0:
            print(f"  Warning: exit code {result.returncode}", file=sys.stderr)

        m = parse_map_from_output(output)
        print(f"  mAP@0.5: {m:.4f}")
        if m > best_map:
            best_map = m
            best_config = (conf, iou)
            print(f"  -> new best")

    print("\nBEST RESULT")
    print("  mAP@0.5:", best_map)
    print("  confidence:", best_config[0] if best_config else "N/A")
    print("  nms_iou:", best_config[1] if best_config else "N/A")
    print("\nRun inference with:")
    print(f"  --confidence {best_config[0]} --nms-iou {best_config[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
