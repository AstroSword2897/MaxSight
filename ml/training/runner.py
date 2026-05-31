"""Shared training builder: ResolvedTrainingConfig -> ProductionTrainLoop.

Local CLI (scripts/ops/train_maxsight.py) and SageMaker container entry
(ml/training/sagemaker_entry.py) both call run_training(resolved) so the
two execution paths stay byte-for-byte identical. Anything that changes
the training graph must flow through ResolvedTrainingConfig.
"""

from __future__ import annotations

import json
import logging
import random
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import Module

from ml.data.data_pipeline import create_data_loaders_for_resolved
from ml.data.dataset_registry import default_registry_path, load_registry
from ml.models.maxsight_cnn import TierConfig, create_model
from ml.training.losses import (
    BoxRegressionLoss,
    ClassificationLoss,
    DistanceZoneLoss,
    MultiHeadLoss,
    ObjectnessLoss,
    ScalarMSELoss,
    UrgencyLoss,
)
from ml.training.run_config import ResolvedTrainingConfig
from ml.training.task_balancing import GradNormMultiHeadLoss
from ml.training.train_loop import ProductionTrainLoop, write_atomic_torch

logger = logging.getLogger(__name__)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU")
        return "cpu"
    return requested


def _build_loss(resolved: ResolvedTrainingConfig) -> Module:
    """Build loss with weights pulled directly from the resolved config.

    Active heads must match resolved.loss.loss_weights exactly; the schema
    validator enforces this, so any KeyError here is a schema bug, not a
    silent drop.
    """
    cfg_loss = resolved.loss
    cfg_model = resolved.model
    weights = dict(cfg_loss.loss_weights)
    loss_functions: dict[str, Module] = {
        "objectness": ObjectnessLoss(),
        "classification": ClassificationLoss(num_classes=cfg_model.num_classes),
        "box": BoxRegressionLoss(),
        "distance": DistanceZoneLoss(),
        "urgency": UrgencyLoss(),
    }
    if cfg_loss.temporal_supervision:
        loss_functions["temporal_consistency"] = ScalarMSELoss()
        loss_functions["flicker"] = ScalarMSELoss()

    declared = set(weights.keys())
    actual = set(loss_functions.keys())
    if declared != actual:
        raise ValueError(
            f"Loss weight keys {sorted(declared)} do not match active loss "
            f"functions {sorted(actual)}; schema validation should have "
            "rejected this earlier."
        )

    if cfg_loss.use_gradnorm:
        return GradNormMultiHeadLoss(
            loss_functions,
            alpha=cfg_loss.gradnorm_alpha,
            update_interval=cfg_loss.gradnorm_update_interval,
            initial_weights=weights,
        )
    return MultiHeadLoss(loss_functions, loss_weights=weights)


def _write_provenance(checkpoint_dir: Path, resolved: ResolvedTrainingConfig) -> Path:
    """Write the resolved-config snapshot next to checkpoints for audit."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = checkpoint_dir / "resolved_config.json"
    provenance_path.write_text(
        json.dumps(resolved.to_canonical_dict(), sort_keys=True, indent=2, default=str)
    )
    return provenance_path


def _backup_artifacts(best_ckpt: Path, resolved: ResolvedTrainingConfig) -> None:
    backup_dir = Path("backups") / datetime.now().strftime("%Y%m%d_%H%M%S")
    models_dir = backup_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    if best_ckpt.exists():
        shutil.copy2(best_ckpt, models_dir / best_ckpt.name)
    bundle = backup_dir / "code.bundle"
    subprocess.run(
        ["git", "bundle", "create", str(bundle), "--all"],
        cwd=str(Path(__file__).resolve().parents[2]),
        check=False,
        capture_output=True,
    )
    (backup_dir / "metadata.json").write_text(
        json.dumps(
            {
                "timestamp": datetime.now().isoformat(),
                "checkpoint": str(best_ckpt),
                "config_hash": resolved.provenance.config_hash,
                "dataset_id": resolved.dataset.dataset_id,
                "dataset_version": resolved.dataset.dataset_version,
            },
            indent=2,
        )
    )
    (backup_dir / "resolved_config.json").write_text(
        json.dumps(resolved.to_canonical_dict(), sort_keys=True, indent=2, default=str)
    )
    logger.info("backup_completed dir=%s", backup_dir)


def run_training(
    resolved: ResolvedTrainingConfig,
    *,
    resume_from: str | None = None,
    resume_model_only: bool = False,
    use_compile: bool = False,
    use_audio: bool = False,
    backup: bool = False,
    registry_path: Path | None = None,
) -> dict[str, Any]:
    """Build everything from `resolved` and run ProductionTrainLoop.

    Local + SageMaker share this entrypoint; any divergence belongs in
    ResolvedTrainingConfig, never in the runner. Returns the trainer's
    result dict augmented with provenance metadata.
    """
    resolved.log_at_startup(logger)
    _seed_everything(resolved.seed)
    device = _resolve_device(resolved.device)
    logger.info("device=%s", device)

    dist_backend = resolved.distributed.backend
    rank, world_size, local_rank = (0, 1, 0)
    if dist_backend in ("ddp", "fsdp"):
        from ml.training.distributed import init_distributed

        rank, world_size, local_rank = init_distributed()
        if device == "cuda" and torch.cuda.is_available():
            device = f"cuda:{local_rank}"
        logger.info("distributed backend=%s rank=%d world_size=%d", dist_backend, rank, world_size)

    cfg_data = resolved.data
    repo_root = Path(__file__).resolve().parents[2]
    registry = load_registry(registry_path or default_registry_path(repo_root))
    train_loader, val_loader, _ = create_data_loaders_for_resolved(
        resolved,
        registry=registry,
        repo_root=repo_root,
        device=device,
    )
    n_train = len(train_loader.dataset) if getattr(train_loader, "dataset", None) is not None else 0
    n_val = len(val_loader.dataset) if getattr(val_loader, "dataset", None) is not None else 0
    logger.info(
        "data train=%d val=%d batch=%d batches=train:%d/val:%d",
        n_train,
        n_val,
        cfg_data.batch_size,
        len(train_loader),
        len(val_loader),
    )

    cfg_model = resolved.model
    cfg_train = resolved.training
    model_section = resolved.to_canonical_dict()["model"]
    tier_config = TierConfig.from_dict(model_section)
    model = create_model(
        num_classes=cfg_model.num_classes,
        use_audio=use_audio,
        condition_mode=cfg_data.condition_mode,
        tier_config=tier_config,
    ).to(device)
    if cfg_train.use_gradient_checkpointing:
        model.use_gradient_checkpointing = True
        logger.info("gradient_checkpointing enabled")
    logger.info("model_params_M=%.2f", sum(p.numel() for p in model.parameters()) / 1e6)

    compile_enabled = use_compile or cfg_train.use_compile
    if compile_enabled and device.startswith("cuda"):
        logger.info("torch.compile enabled")
        model = torch.compile(model)

    if dist_backend == "ddp" and world_size > 1:
        from ml.training.distributed import wrap_ddp

        model = wrap_ddp(model, local_rank)
    elif dist_backend == "fsdp" and world_size > 1:
        from ml.training.distributed import wrap_fsdp

        model = wrap_fsdp(model)

    loss_fn = _build_loss(resolved)

    ckpt_dir = Path(resolved.checkpoint.save_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = _write_provenance(ckpt_dir, resolved)
    logger.info(
        "provenance_written path=%s hash=%s", provenance_path, resolved.provenance.config_hash
    )

    assert isinstance(model, Module)
    trainer = ProductionTrainLoop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        device=device,
        learning_rate=cfg_train.learning_rate,
        weight_decay=cfg_train.weight_decay,
        num_epochs=cfg_train.num_epochs,
        use_mixed_precision=cfg_train.mixed_precision,
        gradient_clip_norm=cfg_train.gradient_clip_norm,
        gradient_accumulation_steps=cfg_train.gradient_accumulation_steps,
        log_interval=resolved.logging_.log_every_n_steps,
        checkpoint_dir=str(ckpt_dir),
        save_best_only=not resolved.checkpoint.save_last,
        freeze_backbone=cfg_train.freeze_backbone,
        freeze_backbone_epochs=cfg_train.freeze_backbone_epochs,
        ema_decay=cfg_train.ema_decay,
        scheduler_type=cfg_train.scheduler_type,
        warmup_epochs=cfg_train.warmup_epochs,
        learning_rate_backbone=cfg_train.learning_rate_backbone,
        learning_rate_head=cfg_train.learning_rate_head,
        num_classes=cfg_model.num_classes,
        checkpoint_interval=cfg_train.checkpoint_interval,
        resume_from=resume_from,
        resume_model_only=resume_model_only,
        seed=resolved.seed,
        early_stopping_patience=cfg_train.early_stopping_patience,
        early_stopping_metric=cfg_train.early_stopping_metric,
        use_gradnorm=resolved.loss.use_gradnorm,
        gradnorm_alpha=resolved.loss.gradnorm_alpha or 1.5,
        gradnorm_update_interval=resolved.loss.gradnorm_update_interval or 100,
    )

    results = trainer.train()
    logger.info(
        "train_done best_path=%s best_val=%.4f",
        results.get("best_model_path"),
        results.get("best_val_loss", float("nan")),
    )

    best_ckpt = Path(results.get("best_model_path", "")) if results.get("best_model_path") else None
    if best_ckpt is not None and best_ckpt.exists():
        _stamp_checkpoint(best_ckpt, resolved)
    if backup and best_ckpt is not None:
        _backup_artifacts(best_ckpt, resolved)

    return {
        **results,
        "provenance_path": str(provenance_path),
        "config_hash": resolved.provenance.config_hash,
    }


def _stamp_checkpoint(path: Path, resolved: ResolvedTrainingConfig) -> None:
    """Embed provenance into the checkpoint and emit a sidecar JSON.

    Sidecar lets ops tooling read provenance without depending on torch.
    """
    sidecar = path.with_suffix(path.suffix + ".provenance.json")
    sidecar.write_text(
        json.dumps(resolved.to_canonical_dict(), sort_keys=True, indent=2, default=str)
    )
    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(str(path), map_location="cpu")
    if not isinstance(ckpt, dict):
        return
    ckpt["provenance"] = resolved.to_canonical_dict()
    ckpt["config_hash"] = resolved.provenance.config_hash
    ckpt["dataset_id"] = resolved.dataset.dataset_id
    ckpt["dataset_version"] = resolved.dataset.dataset_version
    write_atomic_torch(path, ckpt)
    logger.info("checkpoint_stamped path=%s sidecar=%s", path, sidecar)


__all__ = ["run_training"]
