"""
Motion/Flow Head

Outputs optical flow for motion tracking therapy tasks.

Phase 2: Therapy Heads
See docs/therapy_system_implementation_plan.md for implementation details.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class MotionHead(nn.Module):
    """
    Motion/flow head for therapy tasks.
    
    Inputs: temporal features from temporal encoder
    Output: 2-channel motion (u, v) [B, 2, H, W]
    
    Losses:
    - L2 motion loss
    - Smoothness regularizer
    """
    
    def __init__(self, in_channels: int = 128):
        super().__init__()
        self.in_channels = in_channels
        
        # Motion estimation network
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 2, kernel_size=1)  # u, v motion
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()  # Normalize to [-1, 1]
    
    def forward(self, temporal_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Arguments:
            temporal_features: Temporal features [B, C, H, W]
        
        Returns:
            Motion flow [B, 2, H, W] - (u, v) channels
        """
        x = self.relu(self.conv1(temporal_features))
        x = self.relu(self.conv2(x))
        motion = self.tanh(self.conv3(x))  # [B, 2, H, W]
        return motion

