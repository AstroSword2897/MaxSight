# Usage: from ml.data.dataset import MaxSightDataset.

from ml.data.dataset import MaxSightDataset
from ml.data.generate_annotations import generate_annotations_from_coco
from ml.data.data_pipeline import (
    create_data_loaders,
    collate_fn,
    compute_class_weights,
    get_data_info
)
from ml.data.video_panoptic import (
    AdaptiveTemporalConfig,
    VideoSamplingConfig,
    PseudoPanopticQualityConfig,
    build_fixed_stride_windows,
    build_adaptive_windows,
    compute_motion_score,
    motion_to_temporal_window,
    prune_pseudo_segments,
    associate_tracks_multi_frame,
    iter_chunks,
)
from ml.data.video_preprocessing import (
    PanopticSegmenter,
    PreprocessingConfig,
    VideoPanopticPreprocessor,
)
from ml.data.video_manifest import (
    CONTRACT_FIXED_STRIDE_T8,
    MANIFEST_SCHEMA_VERSION,
    validate_manifest_v1,
)
from ml.data.temporal_clip_targets import TemporalClipTargets, derive_temporal_clip_targets
from ml.data.video_clip_dataset import VideoClipManifestDataset

# Production accessibility dataset (therapy-oriented)
# Note: AccessibilityDataset and related functions are available via direct import:.
# From ml.data.create_accessibility_dataset import AccessibilityDataset.
# They are not re-exported here to avoid unused import warnings.

__all__ = [
    'MaxSightDataset',
    'generate_annotations_from_coco',
    'create_data_loaders',
    'collate_fn',
    'compute_class_weights',
    'get_data_info',
    'VideoSamplingConfig',
    'AdaptiveTemporalConfig',
    'PseudoPanopticQualityConfig',
    'build_fixed_stride_windows',
    'build_adaptive_windows',
    'compute_motion_score',
    'motion_to_temporal_window',
    'prune_pseudo_segments',
    'associate_tracks_multi_frame',
    'iter_chunks',
    'PanopticSegmenter',
    'PreprocessingConfig',
    'VideoPanopticPreprocessor',
    'MANIFEST_SCHEMA_VERSION',
    'CONTRACT_FIXED_STRIDE_T8',
    'validate_manifest_v1',
    'TemporalClipTargets',
    'derive_temporal_clip_targets',
    'VideoClipManifestDataset',
]







