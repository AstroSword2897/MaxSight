#!/usr/bin/env python3
"""SageMaker training container entry point for MaxSight.

Single execution path with the local CLI (scripts/ops/train_maxsight.py):
both call ml.training.runner.run_training(resolved). This module just
gathers --config + SM hyperparameter overrides into ResolvedTrainingConfig,
mirrors checkpoints to SM_MODEL_DIR, and writes the metadata SageMaker
expects (model_meta.json, RunTracker artefacts).

SageMaker injects:
  SM_MODEL_DIR         - where to write packaged artefacts
  SM_OUTPUT_DATA_DIR   - where to write logs/reports
  SM_CHANNEL_TRAIN     - training data channel
  SM_CHANNEL_VAL       - validation data channel
  SM_CHANNEL_GOLD      - gold index channel (optional)
  SM_HP_*              - hyperparameter env vars from estimator.fit()
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
from typing import Any, Dict

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ml.infra.experiment_tracker import RunTracker  # noqa: E402
from ml.training.run_config import (  # noqa: E402
    ConfigValidationError,
    ResolvedTrainingConfig,
    _UNSET,
    cli_overrides_from_namespace,
)
from ml.training.runner import run_training  # noqa: E402
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

# Same dotted override map as the local CLI; keeps both entrypoints honest.
_OVERRIDE_MAP = {
    "epochs": "training.num_epochs",
    "lr": "training.learning_rate",
    "weight_decay": "training.weight_decay",
    "warmup_epochs": "training.warmup_epochs",
    "gradient_clip": "training.gradient_clip_norm",
    "fp16": "training.mixed_precision",
    "freeze_backbone": "training.freeze_backbone",
    "freeze_backbone_epochs": "training.freeze_backbone_epochs",
    "batch_size": "data.batch_size",
    "num_workers": "data.num_workers",
    "device": "device",
    "seed": "seed",
    "checkpoint_dir": "checkpoint.save_dir",
    "train_annotation": "data.train_annotation_file",
    "val_annotation": "data.val_annotation_file",
    "image_dir": "data.image_dir",
    "run_id": "run_id",
    "experiment": "experiment",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("MaxSight SageMaker entry point")

    p.add_argument("--config", type=Path, required=True,
                   help="Tier YAML config (mirrored from the launching client)")
    p.add_argument("--gold-index", type=Path, default=None,
                   help="training_index.json; logged for provenance only")

    # Explicit overrides; missing flags = YAML wins.
    p.add_argument("--epochs", type=int, default=_UNSET)
    p.add_argument("--batch-size", type=int, default=_UNSET)
    p.add_argument("--lr", type=float, default=_UNSET)
    p.add_argument("--weight-decay", type=float, default=_UNSET)
    p.add_argument("--warmup-epochs", type=int, default=_UNSET)
    p.add_argument("--seed", type=int, default=_UNSET)
    p.add_argument("--num-workers", type=int, default=_UNSET)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default=_UNSET)
    p.add_argument("--fp16", dest="fp16", action="store_const", const=True, default=_UNSET)
    p.add_argument("--gradient-clip", type=float, default=_UNSET)
    p.add_argument("--freeze-backbone", dest="freeze_backbone", action="store_const", const=True, default=_UNSET)
    p.add_argument("--freeze-backbone-epochs", type=int, default=_UNSET)
    p.add_argument("--checkpoint-dir", type=Path, default=_UNSET)
    p.add_argument("--train-annotation", type=Path, default=_UNSET)
    p.add_argument("--val-annotation", type=Path, default=_UNSET)
    p.add_argument("--image-dir", type=Path, default=_UNSET)

    p.add_argument("--run-id", default=_UNSET)
    p.add_argument("--experiment", default=_UNSET)

    return p.parse_args()


def _hp_overrides_from_env() -> Dict[str, Any]:
    """Pull SM_HP_* env vars and route them through the override map.

    SageMaker exposes hyperparameters as both CLI args and SM_HP_* env vars;
    we accept both, dropping any key not in the override map so the schema
    layer enforces a single shape.
    """
    out: Dict[str, Any] = {}
    for env_key, raw in os.environ.items():
        if not env_key.startswith("SM_HP_"):
            continue
        attr = env_key[len("SM_HP_"):].lower().replace("-", "_")
        dotted = _OVERRIDE_MAP.get(attr)
        if dotted is None:
            continue
        out[dotted] = _coerce_hp(raw)
    return out


def _coerce_hp(value: str) -> Any:
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if "." in value or "e" in value.lower():
            return float(value)
        return int(value)
    except ValueError:
        return value


# ── Training ──────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    cli_overrides = cli_overrides_from_namespace(args, _OVERRIDE_MAP)
    sm_overrides = _hp_overrides_from_env()
    if "run_id" not in cli_overrides:
        cli_overrides["run_id"] = sm_env("TRAINING_JOB_NAME") or f"sm_{time.strftime('%Y%m%d_%H%M%S')}"
    if "experiment" not in cli_overrides:
        cli_overrides["experiment"] = "maxsight"

    try:
        resolved = ResolvedTrainingConfig.from_sources(
            args.config,
            cli_overrides=cli_overrides,
            sm_hyperparameters=sm_overrides,
        )
    except ConfigValidationError as exc:
        logger.error("config_invalid: %s", exc)
        raise SystemExit(2)

    out_dir = output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    with RunTracker(run_id=resolved.run_id, experiment=resolved.experiment) as tracker:
        tracker.log_params({
            "tier": resolved.model.tier,
            "epochs": resolved.training.num_epochs,
            "batch_size": resolved.data.batch_size,
            "lr": resolved.training.learning_rate,
            "weight_decay": resolved.training.weight_decay,
            "device": resolved.device,
            "fp16": resolved.training.mixed_precision,
            "freeze_backbone": resolved.training.freeze_backbone,
            "freeze_backbone_epochs": resolved.training.freeze_backbone_epochs,
            "config_hash": resolved.provenance.config_hash,
            "dataset_id": resolved.dataset.dataset_id,
            "dataset_version": resolved.dataset.dataset_version,
        })

        gold_idx = args.gold_index or (channel_dir("gold") / "training_index.json")
        if gold_idx.exists():
            tracker.log_dataset_provenance(gold_idx)

        results = run_training(
            resolved,
            resume_from=None,
            resume_model_only=False,
            use_compile=False,
            use_audio=False,
            backup=False,
        )

        ckpt_dir = Path(resolved.checkpoint.save_dir)
        best_ckpt = ckpt_dir / "best.pt"
        if not best_ckpt.exists():
            best_path = results.get("best_model_path")
            if best_path:
                best_ckpt = Path(best_path)
        if best_ckpt.exists():
            tracker.log_artefact(best_ckpt, tag="best_checkpoint")
            dest = model_dir() / best_ckpt.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if best_ckpt != dest:
                shutil.copy2(best_ckpt, dest)
                sidecar = best_ckpt.with_suffix(best_ckpt.suffix + ".provenance.json")
                if sidecar.exists():
                    shutil.copy2(sidecar, dest.with_suffix(dest.suffix + ".provenance.json"))

        meta = {
            "run_id": resolved.run_id,
            "experiment": resolved.experiment,
            "tier": resolved.model.tier,
            "epochs_trained": resolved.training.num_epochs,
            "config_hash": resolved.provenance.config_hash,
            "dataset_id": resolved.dataset.dataset_id,
            "dataset_version": resolved.dataset.dataset_version,
            "best_val_map": tracker.best_metric("val_map", mode="max"),
            "best_val_loss": tracker.best_metric("val_loss", mode="min"),
        }
        (model_dir() / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.info("training_complete artifacts=%s", model_dir())


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
