"""
Confidence/Uncertainty Head

Outputs uncertainty scores for model confidence estimation.

Phase 2: Therapy Heads
See docs/therapy_system_implementation_plan.md for implementation details.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class UncertaintyHead(nn.Module):
    """
    Confidence/uncertainty head for therapy tasks.
    
    Inputs: scene embedding
    Outputs:
    - uncertainty_score [B, 1]
    
    Loss:
    - NLL loss
    - ECE calibration loss
    """
    
    def __init__(self, scene_dim: int = 256):
        super().__init__()
        self.scene_dim = scene_dim
        
        # Uncertainty estimation network
        self.fc1 = nn.Linear(scene_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, scene_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Arguments:
            scene_embedding: Scene embedding [B, scene_dim]
        
        Returns:
            Dictionary with:
                - 'uncertainty_score': [B, 1] - Uncertainty [0, 1]
        """
        x = self.relu(self.fc1(scene_embedding))
        x = self.relu(self.fc2(x))
        uncertainty = self.sigmoid(self.fc3(x))  # [B, 1]
        
        return {
            'uncertainty_score': uncertainty
        }

