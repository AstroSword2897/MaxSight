"""Attention modules for MaxSight 3.0."""

from .attention import (
    CBAM,
    ChannelAttention,
    CrossModalAttention,
    CrossTaskAttention,
    SEBlock,
    SpatialAttention,
)

__all__ = [
    "CBAM",
    "SEBlock",
    "ChannelAttention",
    "SpatialAttention",
    "CrossModalAttention",
    "CrossTaskAttention",
]
