#!/usr/bin/env python3
"""Terminal script to improve mAP for all condition models (detection checkpoints) without retraining."""
import argparse
import itertools
import json
import os
import re
import subprocess
import sys
from pathlib import Path

try:
    REPO = Path(__file__).resolve().parents[1]
except NameError:
    REPO = Path.cwd()
sys.path.insert(0, str(REPO))


def _find_checkpoints_base() -> Path | None:
    """Discover base dir that has checkpoints_<condition>/best_model.pt."""
    candidates = [os.environ.get("CHECKPOINTS_BASE")]
    candidates += [REPO / "checkpoints", REPO / "backups"]
    home = Path.home()
    candidates += [home / "Google Drive" / "My Drive" / "MaxSight"]
    candidates += list(home.glob("Library/CloudStorage/GoogleDrive-*/My Drive/MaxSight"))
    candidates += [Path("/content/drive/MyDrive/MaxSight")]
    for base in candidates:
        if not base:
            continue
        base = Path(base)
        if not base.exists():
            continue
        try:
            for d in base.iterdir():
                if d.is_dir() and d.name.startswith("checkpoints_") and (d / "best_model.pt").exists():
                    return base.resolve()
        except OSError:
            continue
    return None

CONDITIONS = [
    "amblyopia", "amd", "astigmatism", "cataracts", "color_blindness",
    "cvi", "diabetic_retinopathy", "glaucoma", "hyperopia", "myopia",
    "presbyopia", "refractive_errors", "retinitis_pigmentosa", "strabismus",
]

CONF_CANDIDATES = [0.3, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]  # Low values for weak/untrained checkpoints.
# Fast grid when targeting mAP (fewer runs): try key conf + auto, two NMS values.
CONF_FAST = [0.1, 0.05, 0.01, 0.001, "auto"]
NMS_IOU_CANDIDATES = [0.5, 0.6, 0.7, 0.8]
NMS_FAST = [0.5, 0.6]


def parse_map_from_output(output: str) -> float:
    """Extract best mAP@0.5 from run_checkpoint_inference log lines."""
    best = -1.0
    for line in output.splitlines():
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
    parser = argparse.ArgumentParser(
        description="Improve mAP for all condition models: sweep confidence/NMS, then run inference with best params."
    )
    parser.add_argument("--checkpoints-base", type=Path, default=None, help="Base dir with checkpoints_<condition>/best_model.pt")
    parser.add_argument("--val-annotation", type=Path, default=REPO / "datasets" / "cleaned_splits" / "maxsight_val.json")
    parser.add_argument("--image-dir", type=Path, default=REPO / "datasets")
    parser.add_argument("--output", type=Path, default=REPO / "inference_data.json")
    parser.add_argument("--config-output", type=Path, default=REPO / "improved_inference_config.json", help="Save best confidence/nms per run")
    parser.add_argument("--conditions", nargs="*", default=None, help="Limit to these conditions (default: all)")
    parser.add_argument("--max-batches", type=int, default=None, help="Cap batches per sweep run (faster sweep)")
    parser.add_argument("--batch-size", type=int, default=64, help="Validation batch size (passed to run_checkpoint_inference)")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (passed to run_checkpoint_inference)")
    parser.add_argument("--skip-sweep", action="store_true", help="Skip sweep; run inference once with --confidence and --nms-iou")
    parser.add_argument("--confidence", type=str, default="0.05", help="Used if --skip-sweep: float or 'auto' for adaptive threshold")
    parser.add_argument("--nms-iou", type=float, default=0.5, help="Used if --skip-sweep")
    parser.add_argument("--target-map", type=float, default=None, metavar="F", help="When sweeping, try to reach at least this mAP@0.5; uses fast grid (fewer conf/nms combos)")
    parser.add_argument("--fast-sweep", action="store_true", help="Use reduced conf/nms grid (fewer runs, faster)")
    parser.add_argument("--quiet", action="store_true", help="Minimal output: only best params and final result.")
    args = parser.parse_args()

    # Resolve checkpoint base.
    base = args.checkpoints_base
    if base is None:
        base = _find_checkpoints_base()
    if base is None:
        base = Path(os.environ.get("CHECKPOINTS_BASE", ""))
    if not base or not base.exists():
        print("No checkpoints base found. Set CHECKPOINTS_BASE or pass --checkpoints-base.", file=sys.stderr)
        return 1
    base = base.resolve()

    val_ann = args.val_annotation.resolve() if args.val_annotation.exists() else args.val_annotation
    image_dir = args.image_dir.resolve() if args.image_dir.exists() else args.image_dir
    script = REPO / "scripts" / "run_checkpoint_inference.py"
    if not script.exists():
        print(f"Not found: {script}", file=sys.stderr)
        return 1

    conditions = args.conditions or CONDITIONS
    try:
        best_conf = float(args.confidence)
    except ValueError:
        best_conf = args.confidence  # E.g. "auto"
    best_nms = args.nms_iou
    best_map = -1.0

    conf_candidates = list(CONF_CANDIDATES)
    if getattr(args, "target_map", None) is not None:
        conf_candidates = list(conf_candidates) + list(CONF_CANDIDATES_TARGET_MAP) + ["auto"]

    if not args.skip_sweep:
        total = len(conf_candidates) * len(NMS_IOU_CANDIDATES)
        n = 0
        for conf, iou in itertools.product(conf_candidates, nms_candidates):
            n += 1
            if not args.quiet:
                print(f"\n[{n}/{total}] confidence={conf}, nms_iou={iou}")
            cmd = [
                sys.executable,
                str(script),
                "--val-annotation", str(val_ann),
                "--image-dir", str(image_dir),
                "--checkpoints-base", str(base),
                "--confidence", str(conf) if not isinstance(conf, str) else conf,
                "--nms-iou", str(iou),
            ]
            if conditions:
                cmd += ["--conditions"] + conditions
            if args.max_batches is not None:
                cmd += ["--max-batches", str(args.max_batches)]
            if args.quiet:
                cmd += ["--quiet"]
            result = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
            output = result.stdout + result.stderr
            if result.returncode != 0 and not args.quiet:
                print(f"  Warning: exit code {result.returncode}", file=sys.stderr)
            m = parse_map_from_output(output)
            if not args.quiet:
                print(f"  mAP@0.5: {m:.4f}")
            if m > best_map:
                best_map = m
                best_conf, best_nms = conf, iou
                if not args.quiet:
                    print("  -> new best")
            if getattr(args, "target_map", None) is not None and best_map >= args.target_map:
                if not args.quiet:
                    print(f"  Reached target mAP {args.target_map}; stopping sweep.")
                break

        if not args.quiet:
            print("\nBEST RESULT")
        print("mAP@0.5:", best_map, "| confidence:", best_conf, "| nms_iou:", best_nms)

    # Full inference with best params (all conditions, full val unless --max-batches)
    if not args.quiet:
        print("\nRunning full inference with best params...")
    cmd = [
        sys.executable,
        str(script),
        "--val-annotation", str(val_ann),
        "--image-dir", str(image_dir),
        "--checkpoints-base", str(base),
        "--confidence", str(best_conf),
        "--nms-iou", str(best_nms),
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--output", str(args.output),
    ]
    if conditions:
        cmd += ["--conditions"] + conditions
    if args.max_batches is not None:
        cmd += ["--max-batches", str(args.max_batches)]
    if args.quiet:
        cmd += ["--quiet"]
    result = subprocess.run(cmd, cwd=str(REPO), capture_output=True if args.quiet else False, text=True)
    if result.returncode != 0:
        if args.quiet and result.stderr:
            print(result.stderr, file=sys.stderr)
        return result.returncode

    config = {
        "checkpoints_base": str(base),
        "best_confidence": best_conf,
        "best_nms_iou": best_nms,
        "inference_data_json": str(args.output),
    }
    with open(args.config_output, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Done. {args.output} | {args.config_output}")
    if best_map <= 0 and not args.quiet:
        print("Note: mAP stayed 0. To reach 0.5 you need trained checkpoints. Run training first, e.g. scripts/train_t5_fast_colab.py or scripts/train_maxsight.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())





