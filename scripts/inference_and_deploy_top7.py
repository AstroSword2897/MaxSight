#!/usr/bin/env python3
"""Run inference and deployment for the top 7 conditions only."""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

try:
    REPO = Path(__file__).resolve().parents[1]
except NameError:
    REPO = Path.cwd()

_SCRIPT = REPO / "scripts" / "inference_and_deploy_top7.py"
if not _SCRIPT.exists() or not (REPO / ".git").exists():
    print("Repo not found: run the clone and cd first (e.g. in Colab run Cell 1 before this script).", file=sys.stderr)
    if Path("/content").exists():
        print("\nColab – run this in a cell first:", file=sys.stderr)
        print("  %cd /content", file=sys.stderr)
        print("  !git clone -q -b feature/multimodal_refactor https://github.com/AstroSword2897/2026-Prototype.git", file=sys.stderr)
        print("  %cd /content/2026-Prototype", file=sys.stderr)
        print("\nThen run:  !python scripts/inference_and_deploy_top7.py ...", file=sys.stderr)
    else:
        print("  cd /path/to/2026-Prototype  then  python scripts/inference_and_deploy_top7.py ...", file=sys.stderr)
    sys.exit(1)

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
    p.add_argument("--top-by-map", action="store_true",
                   help="Deploy the top 7 conditions by mAP (from inference_data.json or run inference over all conditions first)")
    p.add_argument("--inference-data", type=Path, default=None,
                   help="Path to inference_data.json (for --top-by-map; default repo inference_data.json)")
    p.add_argument("--sweep-for-map", action="store_true",
                   help="Sweep confidence/NMS to maximize mAP (slower; aims for higher mAP e.g. 0.5)")
    p.add_argument("--target-map", type=float, default=None, metavar="F",
                   help="When sweeping, extend search to try to reach at least this mAP@0.5 (e.g. 0.5)")
    p.add_argument("--quick", action="store_true", help="Deploy with JIT-only export (faster, skip ExecuTorch)")
    p.add_argument("--quiet", action="store_true", help="Less output")
    args = p.parse_args()

    def _is_placeholder(p: Optional[Path]) -> bool:
        if p is None:
            return True
        s = str(p)
        return "/path/to" in s or s == "/path/to" or "path/to" in s

    if _is_placeholder(args.checkpoints_base):
        args.checkpoints_base = None
    if _is_placeholder(args.output_dir):
        args.output_dir = None
    if _is_placeholder(args.val_annotation):
        args.val_annotation = None
    if _is_placeholder(args.image_dir):
        args.image_dir = None

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
            print("No checkpoints base found.", file=sys.stderr)
            if Path("/content/drive").exists():
                print("On Colab use: --checkpoints-base /content/drive/MyDrive/MaxSight", file=sys.stderr)
                print("(Mount Drive first: from google.colab import drive; drive.mount(\"/content/drive\"))", file=sys.stderr)
            else:
                print("Pass --checkpoints-base <dir> or set CHECKPOINTS_BASE. Discover: python scripts/find_trained_checkpoints.py", file=sys.stderr)
            return 1
    base = base.resolve()
    out_dir = Path(args.output_dir).resolve() if args.output_dir else (base / "exports_top7")
    inference_data_path = Path(args.inference_data or REPO / "inference_data.json").resolve()
    top_by_map = getattr(args, "top_by_map", False)

    run_inference = (not args.skip_inference) and args.val_annotation and args.val_annotation.exists()
    if args.val_annotation and not args.val_annotation.exists():
        print(f"Warning: val-annotation not found {args.val_annotation}; skipping inference.", file=sys.stderr)
        run_inference = False
    if run_inference and not args.image_dir:
        print("Warning: --image-dir required for inference; skipping inference.", file=sys.stderr)
        run_inference = False

    sweep_for_map = getattr(args, "sweep_for_map", False)
    target_map = getattr(args, "target_map", None)

    if top_by_map and not inference_data_path.exists() and run_inference:
        if sweep_for_map:
            print("Step 1/2: Inference (all conditions, fast sweep for best mAP)...")
            cmd = [
                sys.executable,
                str(REPO / "scripts" / "improve_map_all_models.py"),
                "--checkpoints-base", str(base),
                "--val-annotation", str(args.val_annotation),
                "--image-dir", str(args.image_dir),
                "--output", str(inference_data_path),
                "--max-batches", str(args.max_batches),
                "--batch-size", str(args.batch_size),
                "--num-workers", str(args.num_workers),
                "--fast-sweep",
            ]
            if target_map is not None:
                cmd += ["--target-map", str(target_map)]
            if args.quiet:
                cmd.append("--quiet")
            r = subprocess.run(cmd, cwd=str(REPO))
        else:
            print("Step 1/2: Inference (all conditions) to rank by mAP...")
            cmd = [
                sys.executable,
                str(REPO / "scripts" / "run_checkpoint_inference.py"),
                "--checkpoints-base", str(base),
                "--val-annotation", str(args.val_annotation),
                "--image-dir", str(args.image_dir),
                "--output", str(inference_data_path),
                "--max-batches", str(args.max_batches),
                "--batch-size", str(args.batch_size),
                "--num-workers", str(args.num_workers),
                "--confidence", "auto",
            ]
            if args.quiet:
                cmd += ["--quiet"]
            r = subprocess.run(cmd, cwd=str(REPO))
        if r.returncode != 0:
            print("Inference step had errors; continuing to deploy with default top 7.", file=sys.stderr)
        else:
            print("Inference done. Top 7 by mAP will be used for deploy.")
    elif run_inference and not top_by_map:
        print("Step 1/2: Inference (fixed top 7, limited batches)...")
        cmd = [
            sys.executable,
            str(REPO / "scripts" / "improve_map_all_models.py"),
            "--checkpoints-base", str(base),
            "--val-annotation", str(args.val_annotation),
            "--image-dir", str(args.image_dir),
            "--output", str(inference_data_path),
            "--conditions"] + TOP7 + [
            "--max-batches", str(args.max_batches),
            "--batch-size", str(args.batch_size),
            "--num-workers", str(args.num_workers),
        ]
        if sweep_for_map:
            cmd.append("--fast-sweep")
            if target_map is not None:
                cmd += ["--target-map", str(target_map)]
        else:
            cmd += ["--skip-sweep", "--confidence", "auto"]
        if args.quiet:
            cmd.append("--quiet")
        r = subprocess.run(cmd, cwd=str(REPO))
        if r.returncode != 0:
            print("Inference step had errors; continuing to deploy.", file=sys.stderr)
        else:
            print("Inference done.")
    elif top_by_map and inference_data_path.exists():
        if not args.quiet:
            print("Step 1/2: Using existing inference data for top-7-by-mAP ranking.")
    elif not args.skip_inference:
        print("Step 1/2: Skipped (no --val-annotation/--image-dir).")

    if args.skip_deploy:
        print("Deploy skipped (--skip-deploy).")
        return 0

    print("Step 2/2: Deploy (export top 7 to iOS bundles)...")
    try:
        import subprocess as _sp
        r = _sp.run(["git", "rev-parse", "HEAD"], cwd=str(REPO), capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            print(f"  (commit: {r.stdout.strip()[:8]}; run 'git pull' if deploy fails)")
    except Exception:
        pass
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "deploy_top7.py"),
        "--checkpoints-base", str(base),
        "--output-dir", str(out_dir),
    ]
    if top_by_map and inference_data_path.exists():
        cmd += ["--top-by-map", "--inference-data", str(inference_data_path)]
    if getattr(args, "quick", False):
        cmd.append("--quick")
    if args.quiet:
        cmd.append("--quiet")
    r = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
    if r.returncode != 0:
        print("Deploy step failed.", file=sys.stderr)
        if r.stdout:
            print(r.stdout, file=sys.stderr)
        if r.stderr:
            print(r.stderr, file=sys.stderr)
        return r.returncode
    print(f"Done. Bundles: {out_dir}")
    print(f"Manifest: {out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())



