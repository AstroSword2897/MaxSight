"""
Depth/Focus Head

Outputs depth map and near/mid/far classification.

Phase 2: Therapy Heads
See docs/therapy_system_implementation_plan.md for implementation details.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class DepthHead(nn.Module):
    """
    Depth/focus head for therapy tasks.
    
    Inputs: fused FPN + temporal features
    Output:
    - depth map [B, H, W]
    - near/mid/far classification [B, 3]
    
    Losses:
    - Photometric loss
    - Sparse synthetic depth loss
    """
    
    def __init__(self, in_channels: int = 256 + 128, dropout: float = 0.1):  # FPN + temporal
        super().__init__()
        self.in_channels = in_channels
        
        # Depth estimation network (with BatchNorm for efficiency and stability)
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.depth_conv = nn.Conv2d(64, 1, kernel_size=1)  # Depth map
        
        # Depth uncertainty head (properly encapsulated)
        self.uncertainty_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()  # Uncertainty in [0, 1]
        )
        
        # Distance zone classification (with dropout for regularization)
        self.zone_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),  # Better than BatchNorm for 1D features
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 3)  # near, mid, far
            # Raw logits for CrossEntropyLoss (softmax applied in loss)
        )
        
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Arguments:
            features: Fused FPN + temporal features [B, C, H, W]
        
        Returns:
            Dictionary with:
                - 'depth_map': [B, H, W] - Depth map [0, 1]
                - 'uncertainty': [B, H, W] - Depth uncertainty [0, 1]
                - 'zones': [B, 3] - Zone logits (not softmaxed)
        """
        # Efficient forward pass with BatchNorm
        x = self.relu(self.bn1(self.conv1(features)))
        x = self.relu(self.bn2(self.conv2(x)))
        
        # Depth map with safe squeeze
        depth_map = self.sigmoid(self.depth_conv(x))
        if depth_map.shape[1] == 1:
            depth_map = depth_map.view(depth_map.size(0), depth_map.size(2), depth_map.size(3))
        else:
            depth_map = depth_map.squeeze(1)  # [B, H, W]
        
        # Uncertainty (properly encapsulated)
        uncertainty = self.uncertainty_conv(x)
        if uncertainty.shape[1] == 1:
            uncertainty = uncertainty.view(uncertainty.size(0), uncertainty.size(2), uncertainty.size(3))
        else:
            uncertainty = uncertainty.squeeze(1)  # [B, H, W]
        
        # Zone classification (raw logits for CrossEntropyLoss)
        zones = self.zone_head(x)  # [B, 3] - raw logits
        
        return {
            'depth_map': depth_map,
            'uncertainty': uncertainty,  # NEW
            'zones': zones  # Raw logits
        }

