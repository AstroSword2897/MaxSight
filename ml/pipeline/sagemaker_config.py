"""SageMaker configuration contract for production temporal pipeline."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml.data.video_panoptic import (
    AdaptiveTemporalConfig,
    PseudoPanopticQualityConfig,
    VideoSamplingConfig,
)
from ml.data.video_preprocessing import PreprocessingConfig
from ml.training.loss_weighting import TemporalWeightSchedule


@dataclass(frozen=True)
class SageMakerPipelineConfig:
    """Config loaded from SageMaker env + hyperparameters."""

    input_dir: Path
    output_dir: Path
    model_dir: Path
    preprocessing: PreprocessingConfig
    temporal_weight_schedule: TemporalWeightSchedule
    advisory_retrieval_top_k: int = 5

    @staticmethod
    def _resolve_input_dir() -> Path:
        # Training uses SM_CHANNEL_TRAIN; Processing mounts the same logical channel under /opt/ml/processing/.
        env = os.environ.get("SM_CHANNEL_TRAIN", "")
        if env:
            return Path(env)
        proc = Path("/opt/ml/processing/input/train")
        if proc.is_dir():
            return proc
        return Path("/opt/ml/input/data/train")

    @staticmethod
    def _resolve_output_dir() -> Path:
        # SM_OUTPUT_DATA_DIR for training; Processing writes to /opt/ml/processing/output/<channel>.
        env = os.environ.get("SM_OUTPUT_DATA_DIR", "")
        if env:
            return Path(env)
        if Path("/opt/ml/processing").is_dir():
            return Path("/opt/ml/processing/output/train")
        return Path("/opt/ml/output/data")

    @staticmethod
    def _resolve_model_dir() -> Path:
        # Artefact dir for training; optional extra channel on Processing jobs.
        env = os.environ.get("SM_MODEL_DIR", "")
        if env:
            return Path(env)
        if Path("/opt/ml/processing").is_dir():
            return Path("/opt/ml/processing/output/model")
        return Path("/opt/ml/model")

    @staticmethod
    def _read_hyperparameters() -> dict[str, Any]:
        hp_path = Path("/opt/ml/input/config/hyperparameters.json")
        if not hp_path.exists():
            return {}
        try:
            return json.loads(hp_path.read_text())
        except Exception:
            return {}

    @classmethod
    def from_env(cls) -> SageMakerPipelineConfig:
        hp = cls._read_hyperparameters()
        input_dir = cls._resolve_input_dir()
        output_dir = cls._resolve_output_dir()
        model_dir = cls._resolve_model_dir()

        sampling = VideoSamplingConfig(
            temporal_window=int(hp.get("temporal_window", 8)),
            temporal_stride=int(hp.get("temporal_stride", 1)),
            temporal_overlap=int(hp.get("temporal_overlap", 2)),
        )
        quality = PseudoPanopticQualityConfig(
            min_confidence=float(hp.get("min_confidence", 0.45)),
            min_area_pixels=float(hp.get("min_area_pixels", 24.0)),
            min_bbox_width=float(hp.get("min_bbox_width", 2.0)),
            min_bbox_height=float(hp.get("min_bbox_height", 2.0)),
        )
        adaptive = AdaptiveTemporalConfig(
            t_min=int(hp.get("adaptive_t_min", 4)),
            t_max=int(hp.get("adaptive_t_max", 16)),
            smooth_factor=float(hp.get("adaptive_smooth_factor", 0.2)),
            alpha_iou=float(hp.get("adaptive_alpha_iou", 0.5)),
            beta_displacement=float(hp.get("adaptive_beta_displacement", 0.5)),
            overlap_ratio=float(hp.get("adaptive_overlap_ratio", 0.25)),
        )
        preprocessing = PreprocessingConfig(
            sampling=sampling,
            quality=quality,
            chunk_size=int(hp.get("chunk_size", 64)),
            segmentation_workers=int(hp.get("segmentation_workers", 4)),
            temporal_lookback=int(hp.get("temporal_lookback", 2)),
            temporal_iou_threshold=float(hp.get("temporal_iou_threshold", 0.3)),
            enable_frame_jitter=bool(int(hp.get("enable_frame_jitter", 0))),
            enable_speed_perturbation=bool(int(hp.get("enable_speed_perturbation", 0))),
            enable_adaptive_windowing=bool(int(hp.get("enable_adaptive_windowing", 1))),
            adaptive=adaptive,
        )
        schedule = TemporalWeightSchedule(
            start_epoch=int(hp.get("weight_start_epoch", 0)),
            warmup_epochs=int(hp.get("weight_warmup_epochs", 10)),
            start_weight=float(hp.get("weight_start", 0.1)),
            target_weight=float(hp.get("weight_target", 0.6)),
        )
        cfg = cls(
            input_dir=input_dir,
            output_dir=output_dir,
            model_dir=model_dir,
            preprocessing=preprocessing,
            temporal_weight_schedule=schedule,
            advisory_retrieval_top_k=int(hp.get("advisory_retrieval_top_k", 5)),
        )
        cfg.preprocessing.validate()
        cfg.temporal_weight_schedule.validate()
        return cfg
