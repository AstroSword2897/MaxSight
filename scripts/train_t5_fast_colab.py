#!/usr/bin/env python3
"""
T5 Fast Training Script for Colab (~4 hour runs)

Data scientist approach: stratified subset for fast iteration (5 warmup + 50 epochs in ~4.3h),
full validation for honest metrics, FP32 for precision, checkpoints every epoch for resume.

Use this for:
- Quick T5 iteration on a subset of data (8% of COCO train = ~9k samples)
- Testing T5 training pipeline end-to-end
- Accumulating progress across Colab sessions via resume (checkpoints every epoch)

Estimated time on A100 (40GB): ~4.3 hours for 55 epochs (4.7 min/epoch)
- Training: ~3.2 min/epoch (9k samples, batch 8, FP32)
- Validation: ~1.0 min/epoch (5k samples, full val set)
- Checkpoint: ~0.5 min/epoch (save to Drive)

For final training on full data, use train_maxsight.py with --epochs 150+.
"""

import argparse
import sys
import random
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

# Project path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.models.maxsight_cnn import create_model, TierConfig, CapabilityTier, COCO_CLASSES
from ml.training.losses import MultiHeadLoss, ObjectnessLoss, ClassificationLoss, BoxRegressionLoss, DistanceZoneLoss, UrgencyLoss
from ml.training.task_balancing import GradNormMultiHeadLoss
from ml.training.train_loop import ProductionTrainLoop
from ml.data.data_pipeline import create_data_loaders
from ml.utils.logging_config import setup_logging

import logging

setup_logging(log_level="INFO", log_dir=Path("logs"))
logger = logging.getLogger(__name__)


def seed_everything(seed: int):
    """Set all seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_loss_fn(num_classes: int, use_gradnorm: bool = True):
    """Create loss function (GradNorm for T5)."""
    base_losses = {
        "objectness": ObjectnessLoss(),
        "classification": ClassificationLoss(num_classes=num_classes),
        "box_regression": BoxRegressionLoss(),
        "distance": DistanceZoneLoss(),
        "urgency": UrgencyLoss(),
    }
    
    if use_gradnorm:
        return GradNormMultiHeadLoss(
            head_losses=base_losses,
            alpha=1.5,
            initial_weights={
                "objectness": 1.0,
                "classification": 1.2,
                "box_regression": 3.0,
                "distance": 0.7,
                "urgency": 1.5,
            },
        )
    else:
        return MultiHeadLoss(
            loss_functions=base_losses,
            loss_weights={
                "objectness": 1.0,
                "classification": 1.2,
                "box_regression": 3.0,
                "distance": 0.7,
                "urgency": 1.5,
            },
        )


def subset_dataset_stratified(dataset, fraction: float, seed: int):
    """
    Create a stratified subset of the dataset (best-effort).
    
    For now: random subset (TODO: add stratification by class if dataset has labels accessible).
    """
    total = len(dataset)
    subset_size = max(1, int(total * fraction))
    
    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total, generator=rng)[:subset_size].tolist()
    
    logger.info(f"Subset: {subset_size}/{total} samples ({fraction*100:.1f}%)")
    return Subset(dataset, indices)


def main():
    parser = argparse.ArgumentParser(description="T5 Fast Training for Colab (2-3 h runs)")
    
    # Data
    parser.add_argument("--data-dir", required=True, help="Data root (COCO dir)")
    parser.add_argument("--checkpoint-dir", default="/content/drive/MyDrive/MaxSight/checkpoints")
    parser.add_argument("--train-annotation", type=Path, required=True, help="Train split JSON")
    parser.add_argument("--val-annotation", type=Path, required=True, help="Val split JSON")
    parser.add_argument("--image-dir", type=Path, default=None)
    
    # Subset (key for ~4 h runs)
    parser.add_argument("--train-fraction", type=float, default=0.08, 
                        help="Fraction of train data (0.08 = 8%%; ~9k samples from COCO; fits 55 epochs in ~4.3 h on A100)")
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="Max train samples (overrides --train-fraction if set)")
    
    # Training (defaults for ~4 h T5 run on A100)
    parser.add_argument("--epochs", type=int, default=55, help="Total epochs (5 warmup + 50 training)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (8-12 for 40GB A100)")
    parser.add_argument("--grad-accumulation-steps", type=int, default=4, 
                        help="Effective batch = batch_size * this (e.g. 8*4=32)")
    parser.add_argument("--learning-rate", type=float, default=7.5e-5, help="T5 default LR")
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers (Colab often 0-2)")
    parser.add_argument("--checkpoint-interval", type=int, default=1, help="Save every N epochs")
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    
    # Hardware
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    
    # Model
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--use-audio", action="store_true")
    parser.add_argument("--condition-mode", choices=[None, "glaucoma", "amd", "cataracts", "color_blindness"], default=None)
    
    # Resume
    parser.add_argument("--resume-from", type=str, default=None, help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    # -----------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------
    seed_everything(args.seed)
    device = args.device
    
    data_dir = Path(args.data_dir)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    image_dir = Path(args.image_dir) if args.image_dir else data_dir
    train_ann = args.train_annotation
    val_ann = args.val_annotation
    
    # Resolve annotation paths (same logic as train_maxsight.py)
    for ann, name in [(train_ann, "train"), (val_ann, "val")]:
        if not ann.exists():
            candidate = data_dir / "cleaned_splits" / ann.name
            if candidate.exists():
                if name == "train":
                    train_ann = candidate
                else:
                    val_ann = candidate
            else:
                candidate2 = data_dir / ann.name
                if candidate2.exists():
                    if name == "train":
                        train_ann = candidate2
                    else:
                        val_ann = candidate2
                else:
                    raise FileNotFoundError(f"{name} annotation not found: {ann}")
    
    if not train_ann.exists() or not val_ann.exists():
        raise FileNotFoundError(f"Annotations missing: train={train_ann}, val={val_ann}")
    
    logger.info(f"Train annotation: {train_ann}")
    logger.info(f"Val annotation: {val_ann}")
    logger.info(f"Image dir: {image_dir}")
    
    # -----------------------------------------------------------------
    # Data (with subset for train)
    # -----------------------------------------------------------------
    logger.info("Creating data loaders...")
    train_loader_full, val_loader, _ = create_data_loaders(
        train_annotation_file=train_ann,
        val_annotation_file=val_ann,
        image_dir=image_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        condition_mode=args.condition_mode,
        apply_lighting_augmentation=True,
    )
    
    # Subset train (data scientist approach: fixed subset for reproducibility)
    train_dataset = train_loader_full.dataset
    total_train = len(train_dataset)  # type: ignore[arg-type]
    
    if args.max_train_samples is not None:
        subset_size = min(args.max_train_samples, total_train)
        fraction = subset_size / total_train
    else:
        fraction = args.train_fraction
        subset_size = max(1, int(total_train * fraction))
    
    train_subset = subset_dataset_stratified(train_dataset, fraction, args.seed)
    
    # Rebuild train loader with subset
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=train_loader_full.collate_fn if hasattr(train_loader_full, 'collate_fn') else None,
    )
    
    logger.info(f"Train: {len(train_subset)}/{total_train} samples ({fraction*100:.1f}%)")
    logger.info(f"Val: {len(val_loader.dataset)} samples (full validation set)")  # type: ignore[arg-type]
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # -----------------------------------------------------------------
    # Model: T5_TEMPORAL
    # -----------------------------------------------------------------
    num_classes = args.num_classes or len(COCO_CLASSES)
    tier_config = TierConfig.for_tier(CapabilityTier.T5_TEMPORAL)
    
    model = create_model(
        num_classes=num_classes,
        use_audio=args.use_audio,
        condition_mode=args.condition_mode,
        tier_config=tier_config,
    ).to(device)
    
    logger.info(f"Model: T5_TEMPORAL, {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    # Loss
    loss_fn = create_loss_fn(num_classes, use_gradnorm=True)
    
    # -----------------------------------------------------------------
    # Trainer
    # -----------------------------------------------------------------
    resume_from = None
    if args.resume_from:
        p = Path(args.resume_from)
        if p.exists():
            resume_from = str(p.resolve())
            logger.info(f"Resuming from: {resume_from}")
        else:
            logger.warning(f"--resume-from: file not found: {args.resume_from}, starting fresh")
    
    from torch.nn import Module
    assert isinstance(model, Module), "model must be nn.Module"
    
    trainer = ProductionTrainLoop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.epochs,
        checkpoint_dir=str(ckpt_dir),
        seed=args.seed,
        use_mixed_precision=False,  # FP32 only
        gradient_clip_norm=args.grad_clip,
        gradient_accumulation_steps=args.grad_accumulation_steps,
        scheduler_type="cosine",
        warmup_epochs=args.warmup_epochs,
        checkpoint_interval=args.checkpoint_interval,
        early_stopping_patience=args.early_stopping_patience,
        resume_from=resume_from,
        use_gradnorm=True,
    )
    
    # -----------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------
    try:
        logger.info(f"Starting T5 training: {args.epochs} epochs, warmup {args.warmup_epochs}, subset {fraction*100:.1f}%")
        results = trainer.train()
        logger.info("Training complete")
        logger.info(f"Best val loss: {results['best_val_loss']:.4f} at epoch {results['best_epoch']}")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
