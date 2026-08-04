"""TimeSformer re-export for temporal long-range attention.

Canonical implementation lives in conv_lstm.py; this module is the public
import path used by TemporalEncoder and integration tests.
"""

from ml.models.temporal.conv_lstm import DividedSpaceTimeAttention, TimeSformer

__all__ = ["TimeSformer", "DividedSpaceTimeAttention"]
