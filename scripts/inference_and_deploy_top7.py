#!/usr/bin/env python3
"""Run inference on the top 7 conditions, then deploy (export) all 7. Target: ~1 hour.

1. Inference: run improve_map_all_models for the 7 (optional, if --val-annotation + --image-dir given; use --max-batches to keep short).
2. Deploy: export each to iOS bundle via deploy_top7.py.

Usage:
  # Deploy only (~35–45 min)
  python scripts/inference_and_deploy_top7.py --checkpoints-base /path/to/MaxSight --output-dir /path/to/MaxSight/exports_top7

  # Inference (quick) + deploy (~1 hr)
  python scripts/inference_and_deploy_top7.py \\
    --checkpoints-base /path/to/MaxSight \\
    --output-dir /path/to/MaxSight/exports_top7 \\
    --val-annotation /path/to/cleaned_splits/maxsight_val.json \\
    --image-dir /path/to/data \\
    --max-batches 10

  # Colab
  python scripts/inference_and_deploy_top7.py \\
    --checkpoints-base /content/drive/MyDrive/MaxSight \\
    --output-dir /content/drive/MyDrive/MaxSight/exports_top7 \\
    --val-annotation /content/drive/MyDrive/MaxSight_Training/cleaned_splits/maxsight_val.json \\
    --image-dir /content/drive/MyDrive/MaxSight_Training \\
    --max-batches 8
"""

import argparse
import subprocess
import sys
from pathlib import Path

try:
    REPO = Path(__file__).resolve().parents[1]
except NameError:
    REPO = Path.cwd()

TOP7 = [
    "amblyopia", "amd", "color_blindness", "cvi", "glaucoma",
    "retinitis_pigmentosa", "strabismus",
]


def main():
    p = argparse.ArgumentParser(
        description="Inference (optional) + deploy for top 7 conditions in ~1 hour."
    )
    p.add_argument("--checkpoints-base", type=Path, default=None,
                   help="Base dir with checkpoints_<cond>/best_model.pt (default: auto-detect or CHECKPOINTS_BASE)")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Deploy output root (default: <checkpoints-base>/exports_top7)")
    p.add_argument("--val-annotation", type=Path, default=None,
                   help="If set, run inference first (COCO-style val JSON)")
    p.add_argument("--image-dir", type=Path, default=None,
                   help="Image root for inference (required if --val-annotation)")
    p.add_argument("--max-batches", type=int, default=10,
                   help="Cap inference batches per condition for speed (default 10)")
    p.add_argument("--skip-inference", action="store_true",
                   help="Skip inference; only deploy")
    p.add_argument("--skip-deploy", action="store_true",
                   help="Skip deploy; only run inference")
    p.add_argument("--quiet", action="store_true", help="Less output")
    args = p.parse_args()

    base = Path(args.checkpoints_base).resolve() if args.checkpoints_base else None
    if base is None:
        r = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "find_trained_checkpoints.py")],
            cwd=str(REPO), capture_output=True, text=True
        )
        if r.returncode == 0 and r.stdout.strip():
            base = Path(r.stdout.strip()).resolve()
        else:
            base = None
        if base is None:
            print("No checkpoints base found. Pass --checkpoints-base /path/to/MaxSight or set CHECKPOINTS_BASE.", file=sys.stderr)
            print("Discover: python scripts/find_trained_checkpoints.py", file=sys.stderr)
            return 1
    base = base.resolve()
    out_dir = Path(args.output_dir).resolve() if args.output_dir else (base / "exports_top7")
    run_inference = (not args.skip_inference) and args.val_annotation and args.val_annotation.exists()
    if args.val_annotation and not args.val_annotation.exists():
        print(f"Warning: val-annotation not found {args.val_annotation}; skipping inference.", file=sys.stderr)
        run_inference = False
    if run_inference and not args.image_dir:
        print("Warning: --image-dir required for inference; skipping inference.", file=sys.stderr)
        run_inference = False

    if run_inference:
        print("Step 1/2: Inference (top 7, limited batches)...")
        cmd = [
            sys.executable,
            str(REPO / "scripts" / "improve_map_all_models.py"),
            "--checkpoints-base", str(base),
            "--val-annotation", str(args.val_annotation),
            "--image-dir", str(args.image_dir),
            "--conditions"] + TOP7 + [
            "--max-batches", str(args.max_batches),
            "--skip-sweep",
            "--confidence", "0.05",
        ]
        if args.quiet:
            cmd.append("--quiet")
        r = subprocess.run(cmd, cwd=str(REPO))
        if r.returncode != 0:
            print("Inference step had errors; continuing to deploy.", file=sys.stderr)
        else:
            print("Inference done.")
    elif not args.skip_inference:
        print("Step 1/2: Skipped (no --val-annotation/--image-dir).")

    if args.skip_deploy:
        print("Deploy skipped (--skip-deploy).")
        return 0

    print("Step 2/2: Deploy (export top 7 to iOS bundles)...")
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "deploy_top7.py"),
        "--checkpoints-base", str(base),
        "--output-dir", str(out_dir),
    ]
    if args.quiet:
        cmd.append("--quiet")
    r = subprocess.run(cmd, cwd=str(REPO))
    if r.returncode != 0:
        print("Deploy step failed.", file=sys.stderr)
        return r.returncode
    print(f"Done. Bundles: {out_dir}")
    print(f"Manifest: {out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
