"""Fusion modules for retrieval."""

from .attention_fusion import AttentionFusion
from .fusion_train import FusionDataset, FusionMLP

# Phase 6: Meta-learning fusion.
try:
    from .meta_fusion import (
        ActiveSceneExploration,
        MetaFusionWeights,
        PredictiveNavigationGuidance,
        UserProfile,
    )

    __all__ = [
        "AttentionFusion",
        "FusionMLP",
        "FusionDataset",
        "MetaFusionWeights",
        "ActiveSceneExploration",
        "PredictiveNavigationGuidance",
        "UserProfile",
    ]
except ImportError:
    __all__ = [
        "AttentionFusion",
        "FusionMLP",
        "FusionDataset",
    ]
