"""
Fatigue/Gaze Head

Outputs fatigue score, blink rate, and fixation stability.

Phase 2: Therapy Heads
See docs/therapy_system_implementation_plan.md for implementation details.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class FatigueHead(nn.Module):
    """
    Fatigue/gaze head for therapy tasks.
    
    Inputs: eye model output + temporal features
    Outputs:
    - fatigue_score [B, 1]
    - blink_rate [B, 1]
    - fixation_stability [B, 1]
    
    Loss:
    - Supervised blink loss
    - Fatigue sequence loss
    """
    
    def __init__(self, eye_dim: int = 4, temporal_dim: int = 128):
        super().__init__()
        self.eye_dim = eye_dim  # blink_prob + fixation + pupil_size
        self.temporal_dim = temporal_dim
        
        # Fatigue estimation network
        self.fc1 = nn.Linear(eye_dim + temporal_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        
        # Output heads
        self.fatigue_head = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()  # Fatigue score [0, 1]
        )
        
        self.blink_rate_head = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()  # Blink rate [0, 1]
        )
        
        self.fixation_stability_head = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()  # Fixation stability [0, 1]
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(
        self,
        eye_features: torch.Tensor,
        temporal_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Arguments:
            eye_features: Eye model features [B, eye_dim] (flattened from eye model outputs)
            temporal_features: Temporal features [B, temporal_dim]
        
        Returns:
            Dictionary with:
                - 'fatigue_score': [B, 1]
                - 'blink_rate': [B, 1]
                - 'fixation_stability': [B, 1]
        """
        # Combine features
        combined = torch.cat([eye_features, temporal_features], dim=1)
        
        x = self.relu(self.fc1(combined))
        x = self.relu(self.fc2(x))
        
        fatigue_score = self.fatigue_head(x)
        blink_rate = self.blink_rate_head(x)
        fixation_stability = self.fixation_stability_head(x)
        
        return {
            'fatigue_score': fatigue_score,
            'blink_rate': blink_rate,
            'fixation_stability': fixation_stability
        }

