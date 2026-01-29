#!/usr/bin/env python3
"""
MaxSight CNN - Full Production Training Script (v2)

Hard guarantees:
- Resume-safe
- Deterministic
- AMP-safe (CUDA only, MPS fallback)
- Backup-safe
- Gradient clipping
- Worker seeding
- Fail-fast dataset validation
"""

import argparse
import sys
import os
import json
import random
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from torch.utils.data import DataLoader

# Project path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.models.maxsight_cnn import create_model
from ml.training.losses import MultiHeadLoss, ObjectnessLoss, ClassificationLoss, BoxRegressionLoss, DistanceZoneLoss, UrgencyLoss
from ml.training.task_balancing import GradNormMultiHeadLoss
from ml.training.train_loop import ProductionTrainLoop
from ml.data.dataset import MaxSightDataset
from ml.utils.logging_config import setup_logging
from ml.models.maxsight_cnn import COCO_CLASSES

import logging

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
setup_logging(log_level="INFO", log_dir=Path("logs"))
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ---------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------
def resolve_device(requested: str) -> str:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA unavailable → CPU fallback")
        return "cpu"
    
    if requested == "mps" and not torch.backends.mps.is_available():
        logger.warning("MPS unavailable → CPU fallback")
        return "cpu"
    
    return requested


# ---------------------------------------------------------------------
# Backup
# ---------------------------------------------------------------------
def backup_training_artifacts(best_ckpt: Path, data_dir: Path):
    backup_dir = Path("backups") / datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # Model
    models_dir = backup_dir / "models"
    models_dir.mkdir()
    shutil.copy2(best_ckpt, models_dir / best_ckpt.name)
    
    # Git bundle
    bundle_path = backup_dir / "code.bundle"
    subprocess.run(
        ["git", "bundle", "create", str(bundle_path), "--all"],
        cwd=Path(__file__).parent.parent,
        check=False,
        capture_output=True,
    )
    
    # Metadata
    meta = {
        "timestamp": datetime.now().isoformat(),
        "data_dir": str(data_dir),
        "checkpoint": str(best_ckpt),
    }
    with open(backup_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    logger.info(f"✅ Backup completed: {backup_dir}")


# ---------------------------------------------------------------------
# Loss function wrapper
# ---------------------------------------------------------------------
def create_loss_fn(num_classes: int, use_gradnorm: bool = False):
    """Create loss function compatible with ProductionTrainLoop."""
    loss_functions = {
        'objectness': ObjectnessLoss(),
        'classification': ClassificationLoss(num_classes=num_classes),
        'box': BoxRegressionLoss(),
        'distance': DistanceZoneLoss(),
        'urgency': UrgencyLoss(),
    }
    
    if use_gradnorm:
        return GradNormMultiHeadLoss(loss_functions)
    else:
        return MultiHeadLoss(loss_functions)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser("Train MaxSight CNN (Production v2)")
    
    # Paths
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint-dir", default="./checkpoints")
    
    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    
    # Hardware
    parser.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default="auto")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 mixed precision (CUDA only)")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile (CUDA only)")
    
    # Resume / backup
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--backup", action="store_true", help="Backup artifacts after training")
    
    # Model
    parser.add_argument("--num-classes", type=int, default=None, help="Number of classes (default: len(COCO_CLASSES))")
    parser.add_argument("--use-audio", action="store_true")
    parser.add_argument("--condition-mode",
                        choices=[None, "glaucoma", "amd", "cataracts", "color_blindness"],
                        default=None)
    
    # Loss
    parser.add_argument("--use-gradnorm", action="store_true", help="Use GradNorm for task balancing")
    
    args = parser.parse_args()
    
    # -----------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------
    device = resolve_device(args.device)
    logger.info(f"Using device: {device}")
    
    set_seed(args.seed)
    
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    data_dir = Path(args.data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(f"train/val directories missing in {data_dir}")
    
    # -----------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------
    train_dataset = MaxSightDataset(train_dir)
    val_dataset = MaxSightDataset(val_dir)
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError(f"Empty dataset detected: train={len(train_dataset)}, val={len(val_dataset)}")
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    g = torch.Generator().manual_seed(args.seed)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        worker_init_fn=seed_worker,
        generator=g,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        worker_init_fn=seed_worker,
        generator=g,
    )
    
    # -----------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------
    num_classes = args.num_classes or len(COCO_CLASSES)
    
    model = create_model(
        num_classes=num_classes,
        use_audio=args.use_audio,
        condition_mode=args.condition_mode,
    ).to(device)
    
    logger.info(f"Model created: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    if args.compile and device == "cuda":
        logger.info("Compiling model with torch.compile...")
        model = torch.compile(model)
    
    loss_fn = create_loss_fn(num_classes, use_gradnorm=args.use_gradnorm)
    
    # -----------------------------------------------------------------
    # Trainer
    # -----------------------------------------------------------------
    # Find latest checkpoint if resuming
    resume_from = None
    if args.resume:
        checkpoints = sorted(ckpt_dir.glob("checkpoint_*.pth"))
        if checkpoints:
            resume_from = str(checkpoints[-1])
            logger.info(f"Resuming from: {resume_from}")
        else:
            logger.warning("--resume specified but no checkpoint found, starting fresh")
    
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
        use_mixed_precision=(args.fp16 and device == "cuda"),  # MPS doesn't support FP16
        gradient_clip_norm=args.grad_clip,
        resume_from=resume_from,
        use_gradnorm=args.use_gradnorm,
    )
    
    # -----------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------
    try:
        results = trainer.train()
        
        logger.info(f"Best model: {results['best_model_path']}")
        logger.info(f"Best val loss: {results['best_val_loss']:.4f}")
        
        # -----------------------------------------------------------------
        # Backup (only if training succeeded)
        # -----------------------------------------------------------------
        if args.backup:
            backup_training_artifacts(
                Path(results["best_model_path"]),
                data_dir,
            )
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        if args.backup:
            logger.warning("Skipping backup due to training failure")
        raise


if __name__ == "__main__":
    main()
