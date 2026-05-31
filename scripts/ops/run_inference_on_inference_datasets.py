#!/usr/bin/env python3
"""Download the two inference datasets (Open Images V6 + ADE20K), then run checkpoint inference on them."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

try:
    REPO = Path(__file__).resolve().parents[1]
except NameError:
    REPO = Path.cwd()
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Default: the 2 datasets we use (Open Images V6 + ADE20K)
DEFAULT_DATASETS = ["open_images_v6", "ade20k"]


def main():
    parser = argparse.ArgumentParser(
        description="Download 2 inference datasets (Open Images V6 + ADE20K) then run checkpoint inference on them."
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download inference datasets first (skips BDD100K; gets open_images_v6 + ade20k)",
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=REPO / "datasets",
        help="Base dir for datasets (default: repo/datasets)",
    )
    parser.add_argument(
        "--checkpoints-base",
        type=Path,
        required=True,
        help="Base dir with checkpoints_<condition>/best_model.pt",
    )
    parser.add_argument(
        "--conditions",
        nargs="*",
        default=None,
        help="Conditions to run (default: cvi amd; use one or more)",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=DEFAULT_DATASETS,
        help=f"Datasets to run inference on (default: {DEFAULT_DATASETS})",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Cap samples per dataset (default: all)"
    )
    parser.add_argument(
        "--confidence", type=float, default=0.3, help="Detection confidence threshold"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output", type=Path, default=None, help="Write per-dataset/condition stats to this JSON"
    )
    parser.add_argument("--quiet", action="store_true", help="Less verbose")
    args = parser.parse_args()

    datasets_dir = Path(args.datasets_dir).resolve()
    checkpoints_base = Path(args.checkpoints_base).resolve()
    conditions = args.conditions or ["cvi", "amd"]
    dataset_names = args.datasets or DEFAULT_DATASETS

    # Step 1: Optional download (Open Images V6 + ADE20K; skip BDD100K)
    if args.download:
        download_script = REPO / "scripts" / "download_inference_datasets.py"
        if not download_script.exists():
            print(f"Not found: {download_script}", file=sys.stderr)
            return 1
        print("Downloading inference datasets (Open Images V6 + ADE20K; BDD100K skipped)...")
        cmd = [
            sys.executable,
            str(download_script),
            "--base-dir",
            str(datasets_dir),
            "--skip-bdd100k",
        ]
        result = subprocess.run(cmd, cwd=str(REPO))
        if result.returncode != 0:
            print("Download step had errors; continuing anyway.", file=sys.stderr)
        else:
            print("Download step finished.")

    # Step 2: Run inference on each dataset with each condition.
    from ml.data.inference_datasets import (
        DetectionPostProcessor,
        create_inference_dataloader,
        run_inference_on_dataset,
    )
    from ml.models.maxsight_cnn import (
        COCO_CLASSES,
        CapabilityTier,
        TierConfig,
        create_model,
    )

    device = args.device
    tier_config = TierConfig.for_tier(CapabilityTier["T5_TEMPORAL"])
    num_classes = len(COCO_CLASSES)
    postprocessor = DetectionPostProcessor(
        confidence_threshold=args.confidence,
        max_detections=20,
        nms_threshold=0.5,
    )

    all_results = {}
    for dname in dataset_names:
        root = datasets_dir / dname
        if not root.exists():
            print(f"Skip {dname}: dir not found {root}", file=sys.stderr)
            all_results[dname] = {"error": f"dir not found: {root}"}
            continue
        if dname.lower() == "open_images_v6":
            val_dir = root / "validation"
            if not val_dir.is_dir():
                err = f"Open Images validation directory not found: {val_dir}"
                print(f"Skip {dname}: {err}", file=sys.stderr)
                all_results[dname] = {"error": err}
                continue
        all_results[dname] = {}
        for cond in conditions:
            ckpt_path = checkpoints_base / f"checkpoints_{cond}" / "best_model.pt"
            if not ckpt_path.exists():
                print(f"Skip {cond}: checkpoint not found {ckpt_path}", file=sys.stderr)
                all_results[dname][cond] = {"error": "checkpoint not found"}
                continue
            if not args.quiet:
                print(f"Running inference: dataset={dname}, condition={cond}")
            try:
                model = create_model(
                    num_classes=num_classes,
                    use_audio=False,
                    condition_mode=cond,
                    tier_config=tier_config,
                )
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
                state = ckpt.get("model_state_dict", ckpt)
                model.load_state_dict(state, strict=False)
                model.to(device)
                try:
                    loader = create_inference_dataloader(
                        dataset_name=dname,
                        root=root,
                        split="validation",
                        batch_size=32,
                        num_workers=0,
                        max_samples=args.max_samples,
                    )
                except FileNotFoundError as fnf:
                    all_results[dname][cond] = {"error": str(fnf)}
                    continue
                results = run_inference_on_dataset(
                    model=model,
                    dataloader=loader,
                    device=device,
                    verbose=not args.quiet,
                    postprocessor=postprocessor,
                    skip_corrupted=True,
                )
                stats = results["stats"]
                all_results[dname][cond] = {
                    "total_images": stats["total_images"],
                    "total_detections": stats["total_detections"],
                    "avg_detections_per_image": round(stats["avg_detections_per_image"], 4),
                    "images_with_detections": stats["images_with_detections"],
                    "images_without_detections": stats["images_without_detections"],
                }
                if not args.quiet:
                    print(
                        f"  {cond}: images={stats['total_images']} detections={stats['total_detections']} avg={stats['avg_detections_per_image']:.2f}"
                    )
            except Exception as e:
                print(f"  {cond} error: {e}", file=sys.stderr)
                all_results[dname][cond] = {"error": str(e)}

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(
                {
                    "datasets_dir": str(datasets_dir),
                    "checkpoints_base": str(checkpoints_base),
                    "results": all_results,
                },
                f,
                indent=2,
            )
        print(f"Wrote {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
