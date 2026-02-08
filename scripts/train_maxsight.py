#!/usr/bin/env python3
"""MaxSight CNN - Full Production Training Script (v2)..."""

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

from ml.models.maxsight_cnn import create_model, TierConfig, CapabilityTier
from ml.training.losses import MultiHeadLoss, ObjectnessLoss, ClassificationLoss, BoxRegressionLoss, DistanceZoneLoss, UrgencyLoss
from ml.training.task_balancing import GradNormMultiHeadLoss
from ml.training.train_loop import ProductionTrainLoop
from ml.data.dataset import MaxSightDataset
from ml.data.data_pipeline import create_data_loaders
from ml.utils.logging_config import setup_logging
from ml.models.maxsight_cnn import COCO_CLASSES

import logging

setup_logging(log_level="INFO", log_dir=Path("logs"))
logger = logging.getLogger(__name__)


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


def resolve_device(requested: str) -> str:
    """Resolve device: auto → cuda if available else cpu; no MPS backend."""
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA unavailable → CPU fallback")
        return "cpu"
    if requested == "mlx":
        logger.info("DEVICE=mlx → using CPU")
        return "cpu"
    return requested


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


def main():
    parser = argparse.ArgumentParser("Train MaxSight CNN (Production v2)")
    
    # Paths
    parser.add_argument("--data-dir", required=True, help="Data root (COCO dir or parent of train/val)")
    parser.add_argument("--checkpoint-dir", default="./checkpoints")
    parser.add_argument("--train-annotation", type=Path, default=None, help="Train split JSON (e.g. datasets/cleaned_splits/maxsight_train.json)")
    parser.add_argument("--val-annotation", type=Path, default=None, help="Val split JSON (e.g. datasets/cleaned_splits/maxsight_val.json)")
    parser.add_argument("--image-dir", type=Path, default=None, help="Image root (default: data-dir; used with train/val-annotation)")
    
    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-3, help="Base LR (head); backbone uses lr * 0.1. Higher can reduce loss faster.")
    parser.add_argument("--weight-decay", type=float, default=5e-5, help="L2 regularization. Slightly lower can allow lower training loss.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--grad-clip", type=float, default=5.0, help="Max gradient norm. 5.0 allows larger steps than 1.0 for faster convergence.")
    parser.add_argument("--grad-accumulation-steps", type=int, default=1, help="Gradient accumulation (effective batch = batch_size * this)")
    parser.add_argument("--scheduler-type", choices=["cosine", "onecycle", "cosine_restarts"], default="cosine")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="LR warmup epochs (e.g. 10%% of 50)")
    parser.add_argument("--lr-backbone", type=float, default=None, help="Learning rate for backbone (default: lr * 0.1)")
    parser.add_argument("--lr-head", type=float, default=None, help="Learning rate for heads (default: --learning-rate)")
    parser.add_argument("--early-stopping-patience", type=int, default=10, help="Stop if no improvement for N epochs (0 = disabled)")
    parser.add_argument("--checkpoint-interval", type=int, default=0, help="Save snapshot every N epochs (0 = only last/best)")
    parser.add_argument(
        "--hyperparameters",
        type=Path,
        default=None,
        help="Path to best_hyperparameters.json from scripts/AutoMLType.py; overrides lr, weight_decay, batch_size, grad_clip for full training with tuned values",
    )
    
    # Hardware
    parser.add_argument("--device", choices=["cpu", "cuda", "mlx", "auto"], default="auto",
                        help="Device: cpu, cuda, mlx (= CPU), or auto (cuda if available)")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile (CUDA only, faster after first epoch)")
    parser.add_argument("--use-amp", action="store_true", help="Use mixed precision (FP16) on CUDA for faster training; default FP32 for stability")
    
    # Resume / backup
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in --checkpoint-dir")
    parser.add_argument("--resume-from", type=str, default=None, metavar="PATH", help="Resume from this checkpoint file (e.g. after copying to another GPU)")
    parser.add_argument("--resume-model-only", action="store_true", help="With --resume-from: load only model + epoch; use current optimizer/scheduler (e.g. new LRs, MLX-style)")
    parser.add_argument("--backup", action="store_true", help="Backup artifacts after training")
    
    # Model
    parser.add_argument("--num-classes", type=int, default=None, help="Number of classes (default: len(COCO_CLASSES))")
    parser.add_argument("--tier", choices=["T5"], default="T5", help="Model tier: T5 (temporal + hybrid + cross-task + cross-modal)")
    parser.add_argument("--use-audio", action="store_true")
    parser.add_argument("--condition-mode",
                        choices=[
                            None, 
                            # Common conditions
                            "glaucoma", "amd", "cataracts", "color_blindness",
                            # Additional conditions
                            "diabetic_retinopathy", "retinitis_pigmentosa", "cvi",
                            # Developmental/alignment
                            "amblyopia", "strabismus",
                            # Refractive errors
                            "refractive_errors", "myopia", "hyperopia", "astigmatism", "presbyopia"
                        ],
                        default=None,
                        help="Vision condition adaptation mode")
    
    # Loss
    parser.add_argument("--use-gradnorm", action="store_true", help="Use GradNorm for task balancing")
    
    args = parser.parse_args()

    # Apply AutoML best hyperparameters if provided
    if args.hyperparameters is not None:
        hp_path = Path(args.hyperparameters)
        if not hp_path.exists():
            raise FileNotFoundError(f"Hyperparameters file not found: {hp_path}")
        with open(hp_path) as f:
            hp_data = json.load(f)
        params = hp_data.get("hyperparameters", hp_data)
        if "learning_rate" in params:
            args.learning_rate = float(params["learning_rate"])
        if "weight_decay" in params:
            args.weight_decay = float(params["weight_decay"])
        if "batch_size" in params:
            args.batch_size = int(params["batch_size"])
        if "gradient_clip_norm" in params:
            args.grad_clip = float(params["gradient_clip_norm"])
        logger.info(
            f"Using tuned hyperparameters from {hp_path}: lr={args.learning_rate}, wd={args.weight_decay}, batch={args.batch_size}, grad_clip={args.grad_clip}"
        )
    
    device = resolve_device(args.device)
    logger.info(f"Using device: {device}")
    
    set_seed(args.seed)
    
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    data_dir = Path(args.data_dir)
    image_dir = args.image_dir or data_dir

    if args.train_annotation and args.val_annotation:
        data_dir_resolved = Path(args.data_dir).resolve()
        train_ann = Path(args.train_annotation).resolve()
        val_ann = Path(args.val_annotation).resolve()
        if not train_ann.exists():
            alt = data_dir_resolved / "cleaned_splits" / train_ann.name
            if alt.exists():
                train_ann = alt
            elif (data_dir_resolved / train_ann.name).exists():
                train_ann = data_dir_resolved / train_ann.name
        if not val_ann.exists():
            alt = data_dir_resolved / "cleaned_splits" / val_ann.name
            if alt.exists():
                val_ann = alt
            elif (data_dir_resolved / val_ann.name).exists():
                val_ann = data_dir_resolved / val_ann.name
        def _resolve_bare(path: Path, name: str) -> Path:
            if path.exists():
                return path
            for candidate_dir in [
                os.environ.get("SPLITS_DIR"),
                "/content/drive/MyDrive/MaxSight_Training/cleaned_splits",
                "/content/drive/MyDrive/MaxSight/datasets/coco_raw/cleaned_splits",
            ]:
                if not candidate_dir:
                    continue
                candidate = Path(candidate_dir) / name
                if candidate.exists():
                    return candidate
            return path

        if not train_ann.exists() or not val_ann.exists():
            train_ann = _resolve_bare(train_ann, train_ann.name)
            val_ann = _resolve_bare(val_ann, val_ann.name)
        if not train_ann.exists() or not val_ann.exists():
            hint = ""
            if train_ann.name.endswith(".json") and len(train_ann.parts) <= 2:
                hint = (
                    " In Colab: run Cell 2 first, then use --train-annotation \"{SPLITS_DIR}/maxsight_train.json\" "
                    "(curly braces so the notebook substitutes the path)."
                )
            raise FileNotFoundError(
                f"Annotation files not found: {train_ann} / {val_ann}.{hint} "
                "Create with: python scripts/gather_training_data.py --coco-dir <coco> --output-dir datasets/cleaned_splits "
                "or use absolute paths (e.g. /content/drive/MyDrive/.../maxsight_train.json)."
            )
        logger.info(f"Using annotations: train={train_ann}, val={val_ann}, image_dir={image_dir}")
        train_loader, val_loader, _ = create_data_loaders(
            train_annotation_file=train_ann,
            val_annotation_file=val_ann,
            test_annotation_file=None,
            image_dir=image_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=(device == "cuda"),
            condition_mode=args.condition_mode,
            apply_lighting_augmentation=True,
        )
        n_train = len(train_loader.dataset) if hasattr(train_loader, "dataset") and train_loader.dataset is not None else 0
        n_val = len(val_loader.dataset) if hasattr(val_loader, "dataset") and val_loader.dataset is not None else 0
        bs = getattr(train_loader, "batch_size", args.batch_size)
        logger.info(
            f"Train samples: {n_train}, Val samples: {n_val}, Batch size: {bs} → Train batches: {len(train_loader)}, Val batches: {len(val_loader)}"
        )
    else:
        train_dir = data_dir / "train"
        val_dir = data_dir / "val"
        if not train_dir.exists() or not val_dir.exists():
            raise FileNotFoundError(
                f"train/val directories missing in {data_dir}. "
                "Use --train-annotation and --val-annotation with paths from scripts/gather_training_data.py or setup_training_data.py."
            )
        train_dataset = MaxSightDataset(train_dir)
        val_dataset = MaxSightDataset(val_dir)
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            raise RuntimeError(
                f"Empty dataset: train={len(train_dataset)}, val={len(val_dataset)}. "
                "Use --train-annotation and --val-annotation pointing to MaxSight JSON splits (e.g. datasets/cleaned_splits/maxsight_train.json)."
            )
        logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
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

    logger.info("Diagnostic check (first train batch)")
    try:
        batch = next(iter(train_loader))
        images = batch.get("images", batch.get("image"))
        targets = {k: v for k, v in batch.items() if k not in ("images", "image")}
        if images is not None and torch.is_tensor(images):
            logger.info(f"Images shape: {images.shape}")
        logger.info(f"Target keys: {list(targets.keys())}")
        for key in ["distance", "urgency", "boxes", "labels", "num_objects"]:
            if key in targets:
                val = targets[key]
                if torch.is_tensor(val):
                    logger.info(f"  {key}: shape={val.shape}, dtype={val.dtype}")
                    if key in ("distance", "urgency"):
                        valid = (val >= 0).sum().item()
                        total = val.numel()
                        logger.info(f"    Valid samples (>=0): {valid}/{total}")
                else:
                    logger.info(f"  {key}: NOT A TENSOR ({type(val).__name__})")
            else:
                logger.info(f"  {key}: MISSING")
    except Exception as e:
        logger.warning(f"Diagnostic check failed: {e}")

    # Model
    num_classes = args.num_classes or len(COCO_CLASSES)
    
    tier = CapabilityTier.T5_TEMPORAL
    tier_config = TierConfig.for_tier(tier)
    
    logger.info(f"Creating model with tier: {args.tier} ({tier.name})")
    logger.info(f"  SE Attention: {tier_config.use_se_attention}")
    logger.info(f"  CBAM Attention: {tier_config.use_cbam_attention}")
    logger.info(f"  Hybrid Backbone (ResNet+ViT): {tier_config.use_hybrid_backbone}")
    logger.info(f"  Dynamic Conv: {tier_config.use_dynamic_conv}")
    logger.info(f"  Cross-Task Attention: {tier_config.use_cross_task_attention}")
    logger.info(f"  Cross-Modal Attention: {tier_config.use_cross_modal_attention}")
    logger.info(f"  Temporal Modeling: {tier_config.use_temporal_modeling}")
    logger.info(f"  Retrieval: {tier_config.use_retrieval}")
    
    model = create_model(
        num_classes=num_classes,
        use_audio=args.use_audio,
        condition_mode=args.condition_mode,
        tier_config=tier_config,
    ).to(device)
    
    logger.info(f"Model created: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    if args.compile and device == "cuda":
        logger.info("Compiling model with torch.compile...")
        model = torch.compile(model)
    
    loss_fn = create_loss_fn(num_classes, use_gradnorm=args.use_gradnorm)
    
    # Find checkpoint to resume from (same machine or after copy to another GPU)
    resume_from = None
    if args.resume_from:
        p = Path(args.resume_from)
        if p.exists():
            resume_from = str(p.resolve())
            logger.info(f"Resuming from: {resume_from}")
        else:
            raise FileNotFoundError(f"--resume-from: file not found: {args.resume_from}")
    elif args.resume:
        for name in ("last_checkpoint.pt", "best_model.pt"):
            c = ckpt_dir / name
            if c.exists():
                resume_from = str(c)
                logger.info(f"Resuming from: {resume_from}")
                break
        if resume_from is None:
            checkpoints = sorted(ckpt_dir.glob("checkpoint_*.pth"))
            if checkpoints:
                resume_from = str(checkpoints[-1])
                logger.info(f"Resuming from: {resume_from}")
        if resume_from is None:
            logger.warning("--resume specified but no checkpoint found, starting fresh")
    
    # Type narrow for trainer (model may be compiled callable)
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
        use_mixed_precision=getattr(args, "use_amp", False),
        gradient_clip_norm=args.grad_clip,
        gradient_accumulation_steps=args.grad_accumulation_steps,
        scheduler_type=args.scheduler_type,
        warmup_epochs=args.warmup_epochs,
        learning_rate_backbone=args.lr_backbone,
        learning_rate_head=args.lr_head,
        checkpoint_interval=args.checkpoint_interval,
        early_stopping_patience=args.early_stopping_patience,
        resume_from=resume_from,
        resume_model_only=args.resume_model_only,
        use_gradnorm=args.use_gradnorm,
    )
    
    try:
        results = trainer.train()
        
        logger.info(f"Best model: {results['best_model_path']}")
        logger.info(f"Best val loss: {results['best_val_loss']:.4f}")
        
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
