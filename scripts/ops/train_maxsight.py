#!/usr/bin/env python3
"""MaxSight CNN - production training entrypoint (config-resolved).

All training-affecting configuration must come from the YAML referenced by
--config; CLI flags here are explicit overrides only and feed through
ResolvedTrainingConfig so local and SageMaker runs share the same
execution graph.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.training.run_config import (
    _UNSET,
    ConfigValidationError,
    ResolvedTrainingConfig,
    cli_overrides_from_namespace,
)
from ml.training.runner import run_training
from ml.utils.logging_config import setup_logging

setup_logging(log_level="INFO", log_dir=Path("logs"))
logger = logging.getLogger(__name__)


# Maps argparse attribute names to dotted ResolvedTrainingConfig paths.
# Anything missing here is intentional: those flags are operational (resume,
# backup, compile) and never affect the training graph.
_OVERRIDE_MAP = {
    "epochs": "training.num_epochs",
    "learning_rate": "training.learning_rate",
    "weight_decay": "training.weight_decay",
    "warmup_epochs": "training.warmup_epochs",
    "grad_clip": "training.gradient_clip_norm",
    "grad_accumulation_steps": "training.gradient_accumulation_steps",
    "scheduler_type": "training.scheduler_type",
    "early_stopping_patience": "training.early_stopping_patience",
    "early_stopping_metric": "training.early_stopping_metric",
    "checkpoint_interval": "training.checkpoint_interval",
    "lr_backbone": "training.learning_rate_backbone",
    "lr_head": "training.learning_rate_head",
    "freeze_backbone": "training.freeze_backbone",
    "freeze_backbone_epochs": "training.freeze_backbone_epochs",
    "use_amp": "training.mixed_precision",
    "batch_size": "data.batch_size",
    "num_workers": "data.num_workers",
    "condition_mode": "data.condition_mode",
    "train_annotation": "data.train_annotation_file",
    "val_annotation": "data.val_annotation_file",
    "image_dir": "data.image_dir",
    "device": "device",
    "seed": "seed",
    "checkpoint_dir": "checkpoint.save_dir",
    "use_gradnorm": "loss.use_gradnorm",
    "temporal_supervision": "loss.temporal_supervision",
    "run_id": "run_id",
    "experiment": "experiment",
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Train MaxSight CNN (config-resolved)")
    p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML config (e.g. ml/training/configs/t5_temporal.yaml)",
    )
    p.add_argument("--run-id", default=_UNSET, help="Override resolved.run_id")
    p.add_argument("--experiment", default=_UNSET, help="Override resolved.experiment")

    p.add_argument("--epochs", type=int, default=_UNSET)
    p.add_argument("--learning-rate", type=float, default=_UNSET)
    p.add_argument("--weight-decay", type=float, default=_UNSET)
    p.add_argument("--seed", type=int, default=_UNSET)
    p.add_argument("--num-workers", type=int, default=_UNSET)
    p.add_argument("--grad-clip", type=float, default=_UNSET)
    p.add_argument("--grad-accumulation-steps", type=int, default=_UNSET)
    p.add_argument(
        "--scheduler-type", choices=["cosine", "onecycle", "cosine_restarts"], default=_UNSET
    )
    p.add_argument("--warmup-epochs", type=int, default=_UNSET)
    p.add_argument("--lr-backbone", type=float, default=_UNSET)
    p.add_argument("--lr-head", type=float, default=_UNSET)
    p.add_argument("--early-stopping-patience", type=int, default=_UNSET)
    p.add_argument("--early-stopping-metric", choices=["val_loss", "val_map"], default=_UNSET)
    p.add_argument("--checkpoint-interval", type=int, default=_UNSET)
    p.add_argument("--batch-size", type=int, default=_UNSET)

    p.add_argument(
        "--freeze-backbone",
        dest="freeze_backbone",
        action="store_const",
        const=True,
        default=_UNSET,
    )
    p.add_argument("--freeze-backbone-epochs", type=int, default=_UNSET)
    p.add_argument(
        "--use-amp",
        dest="use_amp",
        action="store_const",
        const=True,
        default=_UNSET,
        help="Enable mixed precision (training.mixed_precision=true)",
    )
    p.add_argument(
        "--use-gradnorm", dest="use_gradnorm", action="store_const", const=True, default=_UNSET
    )
    p.add_argument(
        "--temporal-supervision",
        dest="temporal_supervision",
        action="store_const",
        const=True,
        default=_UNSET,
    )

    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default=_UNSET)
    p.add_argument("--checkpoint-dir", default=_UNSET, help="Override checkpoint.save_dir")
    p.add_argument("--train-annotation", default=_UNSET, help="Override data.train_annotation_file")
    p.add_argument("--val-annotation", default=_UNSET, help="Override data.val_annotation_file")
    p.add_argument("--image-dir", default=_UNSET, help="Override data.image_dir")
    p.add_argument("--condition-mode", default=_UNSET, help="Override data.condition_mode")

    # Operational flags (do not enter ResolvedTrainingConfig).
    p.add_argument("--resume-from", type=str, default=None)
    p.add_argument("--resume-model-only", action="store_true")
    p.add_argument("--compile", action="store_true", help="torch.compile (CUDA only)")
    p.add_argument("--use-audio", action="store_true")
    p.add_argument("--backup", action="store_true")
    p.add_argument(
        "--print-config",
        action="store_true",
        help="Resolve config + print canonical JSON, then exit (dry-run)",
    )
    p.add_argument(
        "--hyperparameters",
        type=Path,
        default=None,
        help="best_hyperparameters.json from scripts/AutoMLType.py; merged into CLI overrides",
    )
    return p


def _hp_overrides(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Hyperparameters file not found: {path}")
    blob = json.loads(path.read_text())
    params = blob.get("hyperparameters", blob)
    out: dict[str, Any] = {}
    if "learning_rate" in params:
        out["training.learning_rate"] = float(params["learning_rate"])
    if "weight_decay" in params:
        out["training.weight_decay"] = float(params["weight_decay"])
    if "batch_size" in params:
        out["data.batch_size"] = int(params["batch_size"])
    if "gradient_clip_norm" in params:
        out["training.gradient_clip_norm"] = float(params["gradient_clip_norm"])
    return out


def main() -> None:
    args = _build_parser().parse_args()
    cli_overrides = cli_overrides_from_namespace(args, _OVERRIDE_MAP)
    if args.hyperparameters is not None:
        cli_overrides.update(_hp_overrides(args.hyperparameters))
        logger.info(
            "merged_hyperparameters source=%s keys=%s",
            args.hyperparameters,
            sorted(cli_overrides.keys()),
        )

    if "run_id" not in cli_overrides:
        from datetime import datetime, timezone

        cli_overrides["run_id"] = datetime.now(timezone.utc).strftime("local-%Y%m%dT%H%M%SZ")
    if "experiment" not in cli_overrides:
        cli_overrides["experiment"] = "maxsight"

    try:
        resolved = ResolvedTrainingConfig.from_sources(args.config, cli_overrides=cli_overrides)
    except ConfigValidationError as exc:
        logger.error("config_invalid: %s", exc)
        raise SystemExit(2)

    if args.print_config:
        print(json.dumps(resolved.to_canonical_dict(), sort_keys=True, indent=2, default=str))
        return

    try:
        run_training(
            resolved,
            resume_from=args.resume_from,
            resume_model_only=args.resume_model_only,
            use_compile=bool(args.compile),
            use_audio=bool(args.use_audio),
            backup=bool(args.backup),
        )
    except Exception:
        logger.exception("training_failed")
        raise


if __name__ == "__main__":
    main()
