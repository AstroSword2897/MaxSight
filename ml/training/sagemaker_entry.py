#!/usr/bin/env python3
"""SageMaker training container entry point for MaxSight.

SageMaker injects:
  SM_MODEL_DIR         → where to write model.tar.gz artefacts
  SM_OUTPUT_DATA_DIR   → where to write output (logs, reports)
  SM_CHANNEL_TRAIN     → path to training data (gold index or COCO)
  SM_CHANNEL_VAL       → path to validation data
  SM_CHANNEL_GOLD      → path to gold index JSON (if provided)
  SM_HP_*              → hyperparameters passed via --hyperparameters

All hyperparameters are also accepted as CLI args so the script can be
run locally with `python ml/training/sagemaker_entry.py --epochs 30 ...`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

# Ensure repo root is on the path whether run via SageMaker or directly.
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ml.infra.experiment_tracker import RunTracker  # noqa: E402
from ml.utils.logging_config import setup_logging  # noqa: E402

setup_logging(log_level="INFO")
logger = logging.getLogger(__name__)


# ── SageMaker environment helpers ─────────────────────────────────────────────

def sm_env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def model_dir() -> Path:
    return Path(sm_env("SM_MODEL_DIR", str(REPO / "model_output")))


def output_dir() -> Path:
    return Path(sm_env("SM_OUTPUT_DATA_DIR", str(REPO / "output")))


def channel_dir(name: str, fallback: str = "") -> Path:
    env_key = f"SM_CHANNEL_{name.upper()}"
    return Path(sm_env(env_key, fallback or str(REPO / "datasets")))


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("MaxSight SageMaker entry point")

    # Data
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--train-annotation", type=Path, default=None)
    parser.add_argument("--val-annotation", type=Path, default=None)
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--gold-index", type=Path, default=None,
                        help="Path to training_index.json (overrides --data-dir etc.)")

    # Model
    parser.add_argument("--tier", type=str, default="T2_DETECTOR",
                        choices=["T0_BASELINE_CNN", "T1_LIGHTWEIGHT", "T2_DETECTOR",
                                 "T3_MULTI_TASK", "T4_ADVANCED", "T5_TEMPORAL"])
    parser.add_argument("--backbone", type=str, default="resnet50",
                        choices=["resnet50", "resnet34", "hybrid_vit"])
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--freeze-backbone-epochs", type=int, default=5)
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to a tier YAML config file")

    # Training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--fp16", action="store_true", help="Mixed-precision training")
    parser.add_argument("--gradient-clip", type=float, default=1.0)

    # Curriculum
    parser.add_argument("--curriculum", action="store_true",
                        help="Enable phased loss unlocking (curriculum training)")

    # Output
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=Path(sm_env("SM_MODEL_DIR", str(REPO / "model_output"))))
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--experiment", type=str, default="maxsight")

    return parser.parse_args()


# ── Data path resolution ──────────────────────────────────────────────────────

def resolve_data_paths(args: argparse.Namespace):
    """Resolve COCO data paths from gold index or explicit args or SM channels."""
    gold_path = args.gold_index or (channel_dir("gold") / "training_index.json")
    if gold_path.exists():
        logger.info("Loading data paths from gold index: %s", gold_path)
        from ml.data.medallion_layout import load_training_index, resolve_coco_for_train
        index = load_training_index(gold_path)
        data_dir, train_ann, val_ann, image_dir = resolve_coco_for_train(index, REPO)
        return data_dir, train_ann, val_ann, image_dir

    # Fall back to explicit args or SM channel dirs.
    data_dir = args.data_dir or channel_dir("train", str(REPO / "datasets"))
    train_ann = args.train_annotation or (data_dir / "annotations" / "instances_train2017.json")
    val_ann = args.val_annotation or (data_dir / "annotations" / "instances_val2017.json")
    image_dir = args.image_dir or data_dir
    return data_dir, train_ann, val_ann, image_dir


# ── Training ──────────────────────────────────────────────────────────────────

def run_training(args: argparse.Namespace) -> None:
    import torch
    from ml.training.train_loop import ProductionTrainLoop
    from ml.data.data_pipeline import create_data_loaders
    from ml.models.maxsight_cnn import create_model, TierConfig, CapabilityTier

    run_id = args.run_id or f"sm_{time.strftime('%Y%m%d_%H%M%S')}"
    ckpt_dir = args.checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir = output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device_str = args.device
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir, train_ann, val_ann, image_dir = resolve_data_paths(args)
    logger.info("Training data: %s", data_dir)
    logger.info("Train ann:     %s", train_ann)
    logger.info("Val ann:       %s", val_ann)

    with RunTracker(run_id=run_id, experiment=args.experiment) as run:
        run.log_params({
            "tier": args.tier,
            "backbone": args.backbone,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "device": device_str,
            "fp16": args.fp16,
            "freeze_backbone": args.freeze_backbone,
            "freeze_backbone_epochs": args.freeze_backbone_epochs,
            "curriculum": args.curriculum,
        })

        gold_idx = args.gold_index or (channel_dir("gold") / "training_index.json")
        if gold_idx.exists():
            run.log_dataset_provenance(gold_idx)

        # Build loaders via existing pipeline.
        train_loader, val_loader = create_data_loaders(
            data_dir=data_dir,
            train_annotation=train_ann,
            val_annotation=val_ann,
            image_dir=image_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        # Model
        tier_enum = CapabilityTier[args.tier]
        tier_cfg = TierConfig.for_tier(tier_enum)
        model = create_model(tier=tier_enum, config=tier_cfg)

        # Training loop (reuse existing ProductionTrainLoop)
        train_loop = ProductionTrainLoop(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device_str,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup_epochs=args.warmup_epochs,
            gradient_clip=args.gradient_clip,
            checkpoint_dir=ckpt_dir,
            use_amp=args.fp16,
            freeze_backbone=args.freeze_backbone,
            freeze_backbone_epochs=args.freeze_backbone_epochs,
            metric_callback=lambda name, val, step: run.log_metric(name, val, step=step),
        )
        train_loop.run()

        best_ckpt = ckpt_dir / "best.pt"
        if best_ckpt.exists():
            run.log_artefact(best_ckpt, tag="best_checkpoint")

        # Copy best checkpoint to SM_MODEL_DIR so SageMaker packages it.
        dest = model_dir() / "best.pt"
        if best_ckpt.exists() and best_ckpt != dest:
            shutil.copy2(best_ckpt, dest)

        # Write a model metadata file.
        meta = {
            "run_id": run_id,
            "tier": args.tier,
            "backbone": args.backbone,
            "epochs_trained": args.epochs,
            "best_val_map": run.best_metric("val_map", mode="max"),
            "best_val_loss": run.best_metric("val_loss", mode="min"),
        }
        (model_dir() / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.info("Training complete. Artefacts in: %s", model_dir())


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
