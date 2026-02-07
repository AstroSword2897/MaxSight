#!/usr/bin/env python3
"""Produce inference data from all condition checkpoints.

This script generates inference data only: for each checkpoint it runs inference on the
validation set and writes metrics (mAP, precision, recall, F1) and latency stats to JSON.
All outputs are inference data—no training or checkpoint modification.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.data.data_pipeline import create_data_loaders
from ml.models.maxsight_cnn import (
    COCO_CLASSES,
    CapabilityTier,
    TierConfig,
    create_model,
)
from ml.training.metrics import DetectionMetrics
from ml.training.train_loop import move_targets_to_device, parse_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _discover_conditions(base_dir: Path):
    """Return sorted list of (condition_name, checkpoint_dir) for each checkpoints_*."""
    base_dir = Path(base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"Checkpoints base dir not found: {base_dir}")
    dirs = sorted(
        d for d in base_dir.iterdir()
        if d.is_dir() and d.name.startswith("checkpoints_")
    )
    out = []
    for d in dirs:
        cond = d.name.replace("checkpoints_", "")
        best = d / "best_model.pt"
        if best.exists():
            out.append((cond, d))
        else:
            logger.warning("Skipping %s: no best_model.pt", d.name)
    return out


def run_inference_for_checkpoint(
    checkpoint_path: Path,
    condition: str,
    val_loader,
    device: str,
    num_classes: int,
    tier: str,
    max_batches: Optional[int],
    confidence_threshold: float,
    diagnose: bool = False,
) -> dict:
    """Load one checkpoint, run validation inference, return metrics and latency stats."""
    model = create_model(
        num_classes=num_classes,
        use_audio=False,
        condition_mode=condition,
        tier_config=TierConfig.for_tier(CapabilityTier[tier]),
    )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state = ckpt.get("model_state_dict", ckpt)
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        logger.debug(
            "Load strict=False: missing=%s unexpected=%s",
            len(result.missing_keys),
            len(result.unexpected_keys),
        )
    model.to(device)
    model.eval()

    detection_metrics = DetectionMetrics(
        num_classes=num_classes,
        iou_thresholds=[0.5, 0.75],
        device=torch.device(device),
        image_size=(224, 224),
        store_predictions=True,
    )
    latencies_ms = []
    detections_per_image = []
    num_images_done = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            try:
                images, targets = parse_batch(batch)
            except Exception as e:
                logger.warning("Skip batch %s: %s", batch_idx, e)
                continue
            images = images.to(device)
            targets = move_targets_to_device(targets, device)
            batch_size = images.shape[0]

            t0 = time.perf_counter()
            outputs = model(images)
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

            if diagnose and batch_idx == 0 and "objectness" in outputs:
                obj = outputs["objectness"].float().flatten()
                obj_np = obj.cpu().numpy()
                n = len(obj_np)
                logger.info(
                    "Diagnostic (first batch): objectness min=%.4f max=%.4f mean=%.4f p50=%.4f p90=%.4f p95=%.4f",
                    float(obj_np.min()), float(obj_np.max()), float(obj_np.mean()),
                    float(np.percentile(obj_np, 50)),
                    float(np.percentile(obj_np, 90)),
                    float(np.percentile(obj_np, 95)),
                )
                for thresh in (0.05, 0.1, 0.2, 0.3):
                    count = (obj_np > thresh).sum()
                    logger.info("  objectness > %.2f: %d / %d (%.1f%%)", thresh, int(count), n, 100.0 * count / n if n else 0)

            try:
                batch_detections = model.get_detections(
                    outputs,
                    confidence_threshold=confidence_threshold,
                )
            except Exception as e:
                logger.warning("get_detections failed batch %s: %s", batch_idx, e)
                continue

            for b in range(batch_size):
                num_images_done += 1
                n_det = len(batch_detections[b]) if b < len(batch_detections) else 0
                detections_per_image.append(n_det)

                gt_boxes_b = targets["boxes"][b]
                gt_labels_b = targets["labels"][b]
                num_objects = int(
                    targets.get("num_objects", torch.tensor([10]))[b].item()
                )
                if num_objects == 0:
                    continue
                gt_boxes_valid = gt_boxes_b[:num_objects].to(device)
                gt_labels_valid = gt_labels_b[:num_objects].to(device)
                if gt_boxes_valid.numel() == 0:
                    continue

                has_pred = b < len(batch_detections) and batch_detections[b]
                if has_pred:
                    pred_boxes_list = []
                    pred_labels_list = []
                    pred_scores_list = []
                    for det in batch_detections[b]:
                        box = det.get("box")
                        if isinstance(box, (list, tuple)) and len(box) == 4:
                            pred_boxes_list.append(box)
                            pred_labels_list.append(
                                det.get("class", det.get("class_id", 0))
                            )
                            pred_scores_list.append(
                                det.get("confidence", 0.5)
                            )
                    if pred_boxes_list:
                        pred_boxes = torch.tensor(
                            pred_boxes_list, device=device, dtype=torch.float32
                        )
                        pred_labels = torch.tensor(
                            pred_labels_list, device=device, dtype=torch.long
                        )
                        pred_scores = torch.tensor(
                            pred_scores_list, device=device, dtype=torch.float32
                        )
                        detection_metrics.update(
                            pred_boxes=pred_boxes,
                            pred_labels=pred_labels,
                            pred_scores=pred_scores,
                            gt_boxes=gt_boxes_valid,
                            gt_labels=gt_labels_valid,
                            iou_threshold=0.5,
                        )
                    else:
                        detection_metrics.update(
                            pred_boxes=torch.empty(0, 4, device=device, dtype=torch.float32),
                            pred_labels=torch.empty(0, device=device, dtype=torch.long),
                            pred_scores=torch.empty(0, device=device, dtype=torch.float32),
                            gt_boxes=gt_boxes_valid,
                            gt_labels=gt_labels_valid,
                            iou_threshold=0.5,
                        )
                else:
                    detection_metrics.update(
                        pred_boxes=torch.empty(0, 4, device=device, dtype=torch.float32),
                        pred_labels=torch.empty(0, device=device, dtype=torch.long),
                        pred_scores=torch.empty(0, device=device, dtype=torch.float32),
                        gt_boxes=gt_boxes_valid,
                        gt_labels=gt_labels_valid,
                        iou_threshold=0.5,
                    )

    map_results = detection_metrics.compute_map(iou_threshold=0.5)
    map_50 = map_results.get("mAP@0.5", map_results.get("mAP", 0.0))
    map_75 = map_results.get("mAP@0.75", 0.0)
    overall_map = map_results.get("mAP", 0.0)
    precision = detection_metrics.compute_precision()
    recall = detection_metrics.compute_recall()
    f1 = detection_metrics.compute_f1()

    lat_arr = latencies_ms
    mean_latency_ms = sum(lat_arr) / len(lat_arr) if lat_arr else 0.0
    det_arr = detections_per_image
    mean_det_per_image = sum(det_arr) / len(det_arr) if det_arr else 0.0

    return {
        "condition": condition,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "num_images": num_images_done,
        "num_batches": len(latencies_ms),
        "mean_latency_ms": round(mean_latency_ms, 2),
        "min_latency_ms": round(min(lat_arr), 2) if lat_arr else 0.0,
        "max_latency_ms": round(max(lat_arr), 2) if lat_arr else 0.0,
        "mAP": round(float(overall_map), 4),
        "mAP_50": round(float(map_50), 4),
        "mAP_75": round(float(map_75), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "mean_detections_per_image": round(mean_det_per_image, 2),
        "min_detections_per_image": min(det_arr) if det_arr else 0,
        "max_detections_per_image": max(det_arr) if det_arr else 0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with all condition checkpoints and save metrics to JSON."
    )
    parser.add_argument(
        "--checkpoints-base",
        type=Path,
        default=Path("/content/drive/MyDrive/MaxSight"),
        help="Base directory containing checkpoints_<condition> folders",
    )
    parser.add_argument(
        "--train-annotation",
        type=Path,
        default=None,
        help="Path to train JSON (used only to create loaders; can use val for inference-only)",
    )
    parser.add_argument(
        "--val-annotation",
        type=Path,
        required=True,
        help="Path to validation annotations JSON",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Image root (e.g. /content/drive/MyDrive/MaxSight_Training)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("inference_data.json"),
        help="Output JSON path for per-checkpoint inference data",
    )
    parser.add_argument(
        "--tier",
        type=str,
        default="T5_TEMPORAL",
        choices=[t.name for t in CapabilityTier],
        help="Model tier (must match trained checkpoints)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=len(COCO_CLASSES),
        help="Number of classes",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Validation batch size",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Cap validation batches per checkpoint (default: all)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.1,
        help="Detection confidence threshold (0.1 recommended for current checkpoints)",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        nargs="*",
        default=None,
        help="Only run these conditions (default: all found)",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Log objectness stats (min/max/mean/percentiles, counts above 0.05/0.1/0.2/0.3) for first batch only",
    )
    args = parser.parse_args()

    train_ann = args.train_annotation or args.val_annotation
    val_ann = Path(args.val_annotation)
    image_dir = Path(args.image_dir)
    if not val_ann.exists():
        raise FileNotFoundError(f"Val annotation not found: {val_ann}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image dir not found: {image_dir}")

    conditions_list = _discover_conditions(args.checkpoints_base)
    if args.conditions:
        conditions_list = [
            (c, d) for c, d in conditions_list
            if c in args.conditions
        ]
    if not conditions_list:
        logger.error("No checkpoints found under %s", args.checkpoints_base)
        return 1

    logger.info("Creating validation loader (shared across checkpoints)...")
    _, val_loader, _ = create_data_loaders(
        train_annotation_file=Path(train_ann),
        val_annotation_file=val_ann,
        test_annotation_file=None,
        image_dir=image_dir,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=(args.device == "cuda"),
        condition_mode=None,
        apply_lighting_augmentation=False,
    )
    num_val = len(val_loader.dataset) if hasattr(val_loader, "dataset") else "?"
    logger.info("Val samples: %s, batches: %s", num_val, len(val_loader))

    results = []
    for condition, ckpt_dir in conditions_list:
        best_path = ckpt_dir / "best_model.pt"
        logger.info("Running inference for condition=%s from %s", condition, best_path)
        try:
            data = run_inference_for_checkpoint(
                checkpoint_path=best_path,
                condition=condition,
                val_loader=val_loader,
                device=args.device,
                num_classes=args.num_classes,
                tier=args.tier,
                max_batches=args.max_batches,
                confidence_threshold=args.confidence,
                diagnose=args.diagnose,
            )
            results.append(data)
            logger.info(
                "  %s: mAP=%.4f mAP@0.5=%.4f prec=%.4f rec=%.4f F1=%.4f mean_latency_ms=%.2f",
                condition,
                data["mAP"],
                data["mAP_50"],
                data["precision"],
                data["recall"],
                data["f1"],
                data["mean_latency_ms"],
            )
        except Exception as e:
            logger.exception("Failed condition %s: %s", condition, e)
            results.append({
                "condition": condition,
                "checkpoint_path": str(best_path),
                "error": str(e),
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_data = {
        "inference_data": True,
        "val_annotation": str(val_ann),
        "image_dir": str(image_dir),
        "tier": args.tier,
        "confidence_threshold": args.confidence,
        "max_batches": args.max_batches,
        "checkpoints_base": str(args.checkpoints_base),
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(out_data, f, indent=2)
    logger.info("Wrote inference data to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
