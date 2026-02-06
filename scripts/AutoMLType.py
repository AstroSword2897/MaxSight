#!/usr/bin/env python3
"""Full AutoML: Optuna-based hyperparameter tuning for MaxSight training...."""

import argparse
import json
import logging
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import optuna
import torch
from torch.utils.data import DataLoader

# Project path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.data.dataset import MaxSightDataset
from ml.data.data_pipeline import create_data_loaders
from ml.models.maxsight_cnn import COCO_CLASSES, create_model
from ml.training.losses import (
    BoxRegressionLoss,
    ClassificationLoss,
    DistanceZoneLoss,
    MultiHeadLoss,
    ObjectnessLoss,
    UrgencyLoss,
)
from ml.training.task_balancing import GradNormMultiHeadLoss
from ml.training.train_loop import ProductionTrainLoop

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def resolve_device(requested: str) -> str:
    if requested == "auto":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA unavailable -> CPU fallback")
        return "cpu"
    if requested == "mps" and (not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available()):
        logger.warning("MPS unavailable -> CPU fallback")
        return "cpu"
    return requested


def create_loss_fn(num_classes: int, use_gradnorm: bool = False):
    """Create loss function compatible with ProductionTrainLoop."""
    loss_functions = {
        "objectness": ObjectnessLoss(),
        "classification": ClassificationLoss(num_classes=num_classes),
        "box": BoxRegressionLoss(),
        "distance": DistanceZoneLoss(),
        "urgency": UrgencyLoss(),
    }
    if use_gradnorm:
        return GradNormMultiHeadLoss(loss_functions)
    return MultiHeadLoss(loss_functions)


def main() -> int:
    parser = argparse.ArgumentParser(description="Optuna hyperparameter tuning for MaxSight")
    parser.add_argument("--data-dir", type=Path, required=True, help="Data root (COCO dir or parent of train/val)")
    parser.add_argument("--train-annotation", type=Path, default=None, help="Train split JSON (e.g. datasets/cleaned_splits/maxsight_train.json); if set, use with --val-annotation and --image-dir")
    parser.add_argument("--val-annotation", type=Path, default=None, help="Val split JSON")
    parser.add_argument("--image-dir", type=Path, default=None, help="Image root (default: data-dir; used with train/val-annotation)")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("./checkpoints_tuning"), help="Base dir for trial checkpoints")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default="cuda", help="Device for training")
    parser.add_argument("--n-trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument("--epochs-per-trial", type=int, default=5, help="Epochs per trial (short for feasible search)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    # FP32 only (--use-fp16 removed)
    parser.add_argument("--use-gradnorm", action="store_true", help="Use GradNorm for task balancing")
    parser.add_argument("--study-name", type=str, default="maxsight_tuning", help="Optuna study name")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL (e.g. sqlite:///optuna.db) for resume")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--num-classes", type=int, default=None, help="Number of classes (default: len(COCO_CLASSES))")
    parser.add_argument("--use-audio", action="store_true")
    parser.add_argument(
        "--condition-mode",
        choices=[None, "glaucoma", "amd", "cataracts", "color_blindness"],
        default=None,
    )
    parser.add_argument(
        "--full-train-after",
        type=int,
        metavar="EPOCHS",
        default=None,
        help="After tuning, run full training with best hyperparameters for EPOCHS (calls train_maxsight.py with --hyperparameters)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    use_annotation_based = args.train_annotation is not None and args.val_annotation is not None
    if use_annotation_based:
        train_ann = Path(args.train_annotation).resolve()
        val_ann = Path(args.val_annotation).resolve()
        if not train_ann.exists():
            for candidate in [data_dir / "cleaned_splits" / train_ann.name, data_dir / train_ann.name]:
                if candidate.exists():
                    train_ann = candidate
                    break
        if not val_ann.exists():
            for candidate in [data_dir / "cleaned_splits" / val_ann.name, data_dir / val_ann.name]:
                if candidate.exists():
                    val_ann = candidate
                    break
        if not train_ann.exists() or not val_ann.exists():
            logger.error(
                "--train-annotation and --val-annotation files must exist. Checked:\n"
                "  train: %s, %s, %s\n  val:   %s, %s, %s\n"
                "Create them with: python scripts/gather_training_data.py --coco-dir <coco> --output-dir datasets/cleaned_splits\n"
                "Or use absolute paths (e.g. /content/drive/MyDrive/MaxSight/cleaned_splits/maxsight_train.json)",
                Path(args.train_annotation).resolve(),
                data_dir / "cleaned_splits" / Path(args.train_annotation).name,
                data_dir / Path(args.train_annotation).name,
                Path(args.val_annotation).resolve(),
                data_dir / "cleaned_splits" / Path(args.val_annotation).name,
                data_dir / Path(args.val_annotation).name,
            )
            return 1
        args.train_annotation = train_ann
        args.val_annotation = val_ann
        image_dir = args.image_dir or data_dir
    else:
        train_dir = data_dir / "train"
        val_dir = data_dir / "val"
        if not train_dir.exists() or not val_dir.exists():
            logger.error("Either provide --train-annotation and --val-annotation, or ensure train/ and val/ exist under --data-dir")
            return 1

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    num_classes = args.num_classes or len(COCO_CLASSES)
    use_fp16 = False  # FP32 only

    def objective(trial: optuna.Trial) -> float:
        # Reproducibility per trial
        trial_seed = args.seed + trial.number
        set_seed(trial_seed)

        # Sample hyperparameters
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
        batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])  # Reduced for T4 GPU stability
        gradient_clip_norm = trial.suggest_float("gradient_clip_norm", 0.5, 2.0)

        # Per-trial checkpoint subdir so trials do not overwrite each other
        trial_ckpt = args.checkpoint_dir / f"trial_{trial.number}"
        trial_ckpt.mkdir(parents=True, exist_ok=True)

        # Datasets and loaders (same pattern as train_maxsight)
        if use_annotation_based:
            train_loader, val_loader, _ = create_data_loaders(
                train_annotation_file=args.train_annotation,
                val_annotation_file=args.val_annotation,
                test_annotation_file=None,
                image_dir=image_dir,
                batch_size=batch_size,
                num_workers=args.num_workers,
                pin_memory=(device == "cuda"),
                condition_mode=args.condition_mode,
                apply_lighting_augmentation=True,
            )
            if len(train_loader.dataset) == 0 or len(val_loader.dataset) == 0:
                raise RuntimeError("Empty dataset")
        else:
            train_dir = data_dir / "train"
            val_dir = data_dir / "val"
            train_dataset = MaxSightDataset(train_dir)
            val_dataset = MaxSightDataset(val_dir)
            if len(train_dataset) == 0 or len(val_dataset) == 0:
                raise RuntimeError("Empty dataset")
            g = torch.Generator().manual_seed(trial_seed)
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=(device == "cuda"),
                worker_init_fn=seed_worker,
                generator=g,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=(device == "cuda"),
                worker_init_fn=seed_worker,
                generator=g,
            )

        model = create_model(
            num_classes=num_classes,
            use_audio=args.use_audio,
            condition_mode=args.condition_mode,
        ).to(device)

        loss_fn = create_loss_fn(num_classes, use_gradnorm=args.use_gradnorm)

        trainer = ProductionTrainLoop(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_epochs=args.epochs_per_trial,
            checkpoint_dir=str(trial_ckpt),
            seed=trial_seed,
            use_mixed_precision=use_fp16,
            gradient_clip_norm=gradient_clip_norm,
            resume_from=None,
            use_gradnorm=args.use_gradnorm,
            checkpoint_interval=0,
        )

        try:
            results = trainer.train()
            return float(results["best_val_loss"])
        except Exception as e:
            logger.warning(f"Trial {trial.number} failed: {e}")
            return float("inf")

    # Create study and run
    load_if_exists = bool(args.storage)
    study = optuna.create_study(
        direction="minimize",
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=load_if_exists,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    # Save best hyperparameters
    best_params = study.best_params
    best_value = study.best_value
    best_trial_number = study.best_trial.number
    out_path = args.checkpoint_dir / "best_hyperparameters.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "best_val_loss": best_value,
                "best_trial_number": best_trial_number,
                "hyperparameters": best_params,
                "n_trials": args.n_trials,
                "epochs_per_trial": args.epochs_per_trial,
            },
            f,
            indent=2,
        )
    logger.info(f"Best trial: {best_trial_number}, best_val_loss: {best_value:.4f}")
    logger.info(f"Best hyperparameters written to {out_path}")

    # Copy best trial's best_model.pt to canonical path for easy reuse
    best_trial_ckpt = args.checkpoint_dir / f"trial_{best_trial_number}" / "best_model.pt"
    if best_trial_ckpt.exists():
        dest = args.checkpoint_dir / "best_model.pt"
        shutil.copy2(best_trial_ckpt, dest)
        logger.info(f"Best model copied to {dest}")

    # Optional: run full training with updated hyperparameters
    if args.full_train_after is not None:
        import subprocess
        hp_path = args.checkpoint_dir / "best_hyperparameters.json"
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "train_maxsight.py"),
            "--data-dir", str(args.data_dir),
            "--checkpoint-dir", str(args.checkpoint_dir / "full_train"),
            "--hyperparameters", str(hp_path),
            "--epochs", str(args.full_train_after),
            "--device", args.device,
            "--num-workers", str(args.num_workers),
        ]
        if use_annotation_based:
            cmd.extend(["--train-annotation", str(args.train_annotation), "--val-annotation", str(args.val_annotation)])
            if args.image_dir is not None:
                cmd.extend(["--image-dir", str(args.image_dir)])
        # FP32 only (no --fp16)
        if args.use_gradnorm:
            cmd.append("--use-gradnorm")
        if args.use_audio:
            cmd.append("--use-audio")
        if args.condition_mode:
            cmd.extend(["--condition-mode", args.condition_mode])
        logger.info("Running full training with best hyperparameters: %s", " ".join(cmd))
        rc = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
        return rc.returncode

    return 0


if __name__ == "__main__":
    sys.exit(main())
