"""Fusion modules for MaxSight 3.0."""

from .multimodal_fusion import (
    EnhancedAudioEncoder,
    HapticEmbedding,
    HapticVisualAttention,
    MultimodalFusion,
    SpatialSoundMapping,
)

__all__ = [
    "MultimodalFusion",
    "EnhancedAudioEncoder",
    "SpatialSoundMapping",
    "HapticEmbedding",
    "HapticVisualAttention",
]
