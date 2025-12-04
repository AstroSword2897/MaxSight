"""
Contrast Map Head

Outputs per-pixel contrast map for therapy tasks.

Phase 2: Therapy Heads
See docs/therapy_system_implementation_plan.md for implementation details.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ContrastMapHead(nn.Module):
    """
    Contrast map head for therapy tasks.
    
    Inputs: det_feats (detection features from FPN)
    Output: [B, H, W] contrast map
    
    Losses:
    - L1 contrast loss
    - Edge-aware loss
    """
    
    def __init__(self, in_channels: int = 256):
        super().__init__()
        self.in_channels = in_channels
        
        # Contrast estimation network
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=1)  # Single channel contrast map
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            features: Detection features [B, C, H, W]
        
        Returns:
            Contrast map [B, H, W]
        """
        x = self.relu(self.conv1(features))
        x = self.relu(self.conv2(x))
        contrast_map = torch.sigmoid(self.conv3(x)).squeeze(1)  # [B, H, W]
        return contrast_map

