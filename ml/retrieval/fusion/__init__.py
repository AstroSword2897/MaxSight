"""Fusion modules for retrieval."""

from .attention_fusion import AttentionFusion
from .fusion_train import FusionMLP, FusionDataset

# Phase 6: Meta-learning fusion.
try:
    from .meta_fusion import (
        MetaFusionWeights,
        ActiveSceneExploration,
        PredictiveNavigationGuidance,
        UserProfile
    )
    __all__ = [
        'AttentionFusion',
        'FusionMLP',
        'FusionDataset',
        'MetaFusionWeights',
        'ActiveSceneExploration',
        'PredictiveNavigationGuidance',
        'UserProfile',
    ]
except ImportError:
    __all__ = [
        'AttentionFusion',
        'FusionMLP',
        'FusionDataset',
    ]








