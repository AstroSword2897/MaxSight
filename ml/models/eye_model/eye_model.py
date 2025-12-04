"""
Eye/Face Micro-Model

Tiny CNN for eye tracking and fatigue detection:
- Blink probability
- Fixation vs saccade patterns
- Pupil-size proxy

Phase 1: Core ML Backbone & Preprocessing
See docs/therapy_system_implementation_plan.md for implementation details.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any


class EyeModel(nn.Module):
    """
    Tiny CNN for eye/face tracking and fatigue detection.
    
    Architecture:
    Conv -> ReLU -> Conv -> ReLU -> FC -> outputs:
    - Blink probability (0–1)
    - Fixation vs saccade pattern
    - Pupil-size proxy
    
    Input: [B, 3, 64, 64] - Face/eye region (64x64 for speed)
    Output: Dict with blink_prob, fixation_pattern, pupil_size
    """
    
    def __init__(self, input_size: Tuple[int, int] = (64, 64)):
        super().__init__()
        self.input_size = input_size
        
        # Tiny CNN architecture
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Output heads
        self.blink_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Blink probability [0, 1]
        )
        
        self.fixation_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),  # [fixation_prob, saccade_prob]
            nn.Softmax(dim=1)
        )
        
        self.pupil_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Pupil size proxy [0, 1]
        )
    
    def forward(self, face_region: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through eye model.
        
        Args:
            face_region: Face/eye region [B, 3, 64, 64]
        
        Returns:
            Dictionary with:
                - 'blink_prob': [B, 1] - Blink probability
                - 'fixation': [B, 2] - [fixation_prob, saccade_prob]
                - 'pupil_size': [B, 1] - Pupil size proxy
        """
        # Feature extraction
        x = F.relu(self.bn1(self.conv1(face_region)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).flatten(1)  # [B, 32]
        
        # Output heads
        blink_prob = self.blink_head(x)
        fixation = self.fixation_head(x)
        pupil_size = self.pupil_head(x)
        
        return {
            'blink_prob': blink_prob,
            'fixation': fixation,
            'pupil_size': pupil_size
        }

