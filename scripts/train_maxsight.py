#!/usr/bin/env python3
"""
MaxSight CNN - Full Production Training Script (Enhanced)

This script handles:
- Argument parsing
- Dataset loading
- Model creation
- Loss & optimizer setup
- FP16 mixed precision (optional)
- Automatic checkpointing & resume
- Full training loop execution

USAGE:
    python scripts/train_maxsight.py \
        --data-dir datasets \
        --epochs 100 \
        --batch-size 32 \
        --device cuda \
        --checkpoint-dir checkpoints
"""

import argparse
import sys
import os
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import random
from torch.utils.data import DataLoader

from ml.models.maxsight_cnn import create_model
from ml.training.losses import MaxSightLoss
from ml.training.train_loop import ProductionTrainLoop
from ml.data.dataset import MaxSightDataset


# Logging setup
import logging
from ml.utils.logging_config import setup_logging

# Setup production logging
setup_logging(log_level="INFO", log_dir=Path("logs"))
logger = logging.getLogger(__name__)


# Seeding
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Annotation file helper
def resolve_annotation_file(directory: Path, provided_file: str | None):
    if provided_file:
        f = Path(provided_file)
        if not f.exists():
            raise FileNotFoundError(f"Annotation file '{f}' not found.")
        return f

    # Auto-detect annotations.json
    candidate = directory / "annotations.json"
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        f"No annotation file provided and none found in {directory}."
    )


# Main
def main():
    parser = argparse.ArgumentParser(description="Train MaxSight CNN (Enhanced)")

    # Paths
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--annotation-file", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")

    # Training configuration
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)

    # Hardware
    parser.add_argument("--device", choices=["cuda", "cpu", "mps", "auto"], default="auto",
                        help="Device to use: cuda, mps, cpu, or auto (prefers mps>cuda>cpu)")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision")
    
    # Early stopping
    parser.add_argument("--early-stopping-patience", type=int, default=10, 
                       help="Early stopping patience (0 to disable)")
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0,
                       help="Minimum change to qualify as improvement")
    parser.add_argument("--early-stopping-metric", choices=["val_loss", "val_map"], 
                       default="val_loss", help="Metric to monitor for early stopping")

    # Model settings
    parser.add_argument("--num_classes", type=int, default=48)
    parser.add_argument("--use_audio", action="store_true")
    parser.add_argument("--condition_mode",
                        choices=[None, "glaucoma", "amd", "cataracts", "color_blindness"],
                        default=None)

    args = parser.parse_args()

    # Validate device
    # Resolve device
    device = args.device
    if device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU.")
        device = "cpu"
    elif device == "mps" and not torch.backends.mps.is_available():
        logger.warning("MPS not available, falling back to CPU.")
        device = "cpu"

    logger.info(f"Using device: {device}")
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
    elif device == "mps":
        logger.info("Using Apple MPS (Metal Performance Shaders) backend")

    # Set seed
    set_seed(args.seed)
    logger.info(f"Random seed set to {args.seed}")

    # Create checkpoint directory
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Checkpoint directory: {ckpt_dir}")

    # Load dataset - Use cleaned splits if available
    data_dir = Path(args.data_dir)
    cleaned_splits_dir = data_dir / "cleaned_splits"
    
    # Check if cleaned splits exist, otherwise fall back to original splits
    if cleaned_splits_dir.exists():
        logger.info("✅ Using cleaned dataset splits (zero overlap, fixed bboxes)")
        train_ann = cleaned_splits_dir / "train_annotations.json"
        val_ann = cleaned_splits_dir / "val_annotations.json"
        # Use parent directory for images (images should be in datasets/train/images, etc.)
        train_dir = data_dir / "train"  # Images still in original location
        val_dir = data_dir / "val"
        
        if not train_ann.exists() or not val_ann.exists():
            raise FileNotFoundError(f"Cleaned splits not found in {cleaned_splits_dir}")
    else:
        logger.warning("⚠️  Cleaned splits not found, using original splits (may have data leakage!)")
        train_dir = data_dir / "train"
        val_dir = data_dir / "val"
        
        if not train_dir.exists() or not val_dir.exists():
            raise FileNotFoundError("Training and validation directories are required.")
        
        train_ann = resolve_annotation_file(train_dir, args.annotation_file)
        val_ann = resolve_annotation_file(val_dir, args.annotation_file)

    logger.info(f"Training annotations: {train_ann}")
    logger.info(f"Validation annotations: {val_ann}")

    logger.info("Loading datasets...")
    train_dataset = MaxSightDataset(
        data_dir=train_dir,
        annotation_file=train_ann,
        condition_mode=args.condition_mode
    )
    val_dataset = MaxSightDataset(
        data_dir=val_dir,
        annotation_file=val_ann,
        condition_mode=args.condition_mode
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(device == "cuda")
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device == "cuda")
    )

    # Create model
    logger.info("Creating model...")
    model = create_model(
        num_classes=args.num_classes,
        condition_mode=args.condition_mode,
        use_audio=args.use_audio
    )
    model.to(device)

    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

    # Load class weights if available
    class_weights_file = cleaned_splits_dir / "class_weights.json" if cleaned_splits_dir.exists() else None
    class_weights = None
    if class_weights_file and class_weights_file.exists():
        logger.info(f"Loading class weights from {class_weights_file}")
        with open(class_weights_file, 'r') as f:
            weights_data = json.load(f)
        class_weights = weights_data
        logger.info("✅ Class weights loaded for weighted loss")
    else:
        logger.warning("⚠️  Class weights not found - using unweighted loss (may have class imbalance issues)")

    # Loss function with class weights
    loss_fn = MaxSightLoss(
        num_classes=args.num_classes,
        class_weights=class_weights
    )

    # Trainer
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
        use_mixed_precision=args.fp16,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        early_stopping_metric=args.early_stopping_metric
    )

    # Training
    logger.info("Starting training loop...")
    results = trainer.train()

    # Final Summary
    logger.info("Training completed successfully")
    logger.info(f"Best model saved to: {results['best_model_path']}")
    logger.info(f"Best validation loss: {results['best_val_loss']:.4f}")


if __name__ == "__main__":
    main()
