"""Data package. Submodules import torch, so public names are loaded lazily.

Usage: ``from ml.data.dataset import MaxSightDataset`` (or ``ml.data.MaxSightDataset``).
Accessibility dataset helpers stay available via ``ml.data.create_accessibility_dataset``.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

_LAZY_EXPORTS: dict[str, str] = {
    "MaxSightDataset": "ml.data.dataset",
    "generate_annotations_from_coco": "ml.data.generate_annotations",
    "create_data_loaders": "ml.data.data_pipeline",
    "collate_fn": "ml.data.data_pipeline",
    "compute_class_weights": "ml.data.data_pipeline",
    "get_data_info": "ml.data.data_pipeline",
    "VideoSamplingConfig": "ml.data.video_panoptic",
    "AdaptiveTemporalConfig": "ml.data.video_panoptic",
    "PseudoPanopticQualityConfig": "ml.data.video_panoptic",
    "build_fixed_stride_windows": "ml.data.video_panoptic",
    "build_adaptive_windows": "ml.data.video_panoptic",
    "compute_motion_score": "ml.data.video_panoptic",
    "motion_to_temporal_window": "ml.data.video_panoptic",
    "prune_pseudo_segments": "ml.data.video_panoptic",
    "associate_tracks_multi_frame": "ml.data.video_panoptic",
    "iter_chunks": "ml.data.video_panoptic",
    "PanopticSegmenter": "ml.data.video_preprocessing",
    "PreprocessingConfig": "ml.data.video_preprocessing",
    "VideoPanopticPreprocessor": "ml.data.video_preprocessing",
    "MANIFEST_SCHEMA_VERSION": "ml.data.video_manifest",
    "CONTRACT_FIXED_STRIDE_T8": "ml.data.video_manifest",
    "validate_manifest_v1": "ml.data.video_manifest",
    "TemporalClipTargets": "ml.data.temporal_clip_targets",
    "derive_temporal_clip_targets": "ml.data.temporal_clip_targets",
    "VideoClipManifestDataset": "ml.data.video_clip_dataset",
}

if TYPE_CHECKING:  # pragma: no cover - static analysis only
    from ml.data.dataset import MaxSightDataset
    from ml.data.generate_annotations import generate_annotations_from_coco
    from ml.data.data_pipeline import create_data_loaders
    from ml.data.data_pipeline import collate_fn
    from ml.data.data_pipeline import compute_class_weights
    from ml.data.data_pipeline import get_data_info
    from ml.data.video_panoptic import VideoSamplingConfig
    from ml.data.video_panoptic import AdaptiveTemporalConfig
    from ml.data.video_panoptic import PseudoPanopticQualityConfig
    from ml.data.video_panoptic import build_fixed_stride_windows
    from ml.data.video_panoptic import build_adaptive_windows
    from ml.data.video_panoptic import compute_motion_score
    from ml.data.video_panoptic import motion_to_temporal_window
    from ml.data.video_panoptic import prune_pseudo_segments
    from ml.data.video_panoptic import associate_tracks_multi_frame
    from ml.data.video_panoptic import iter_chunks
    from ml.data.video_preprocessing import PanopticSegmenter
    from ml.data.video_preprocessing import PreprocessingConfig
    from ml.data.video_preprocessing import VideoPanopticPreprocessor
    from ml.data.video_manifest import MANIFEST_SCHEMA_VERSION
    from ml.data.video_manifest import CONTRACT_FIXED_STRIDE_T8
    from ml.data.video_manifest import validate_manifest_v1
    from ml.data.temporal_clip_targets import TemporalClipTargets
    from ml.data.temporal_clip_targets import derive_temporal_clip_targets
    from ml.data.video_clip_dataset import VideoClipManifestDataset

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    module = _LAZY_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module), name)
