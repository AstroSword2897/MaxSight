"""
ROI Priority Head

Outputs ROI utility scores for prioritization.

Phase 2: Therapy Heads
See docs/therapy_system_implementation_plan.md for implementation details.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ROIPriorityHead(nn.Module):
    """
    ROI priority head for therapy tasks.
    
    Inputs: scene embedding + region pools
    Output: ROI utility score [B, N] for N regions
    
    Losses:
    - Pairwise ranking loss
    """
    
    def __init__(self, scene_dim: int = 256, roi_dim: int = 256):
        super().__init__()
        self.scene_dim = scene_dim
        self.roi_dim = roi_dim
        
        # Priority scoring network
        self.fc1 = nn.Linear(scene_dim + roi_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Single utility score per ROI
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(
        self,
        scene_embedding: torch.Tensor,
        roi_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            scene_embedding: Scene embedding [B, scene_dim]
            roi_features: ROI features [B, N, roi_dim]
        
        Returns:
            ROI utility scores [B, N]
        """
        B, N, _ = roi_features.shape
        
        # Expand scene embedding to match ROI features
        scene_expanded = scene_embedding.unsqueeze(1).expand(B, N, -1)  # [B, N, scene_dim]
        
        # Concatenate scene + ROI features
        combined = torch.cat([scene_expanded, roi_features], dim=2)  # [B, N, scene_dim + roi_dim]
        
        # Score each ROI
        x = self.relu(self.fc1(combined))
        x = self.relu(self.fc2(x))
        scores = self.sigmoid(self.fc3(x)).squeeze(-1)  # [B, N]
        
        return scores

