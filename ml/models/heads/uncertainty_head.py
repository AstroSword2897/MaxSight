"""
Global Confidence Aggregator for MaxSight 3.0

FIXED: Reframed as Global Confidence Aggregator (not isolated uncertainty head).
Uncertainty isn't a metric — it's a control signal. If uncertainty is high, the system speaks less.

Phase 2: Therapy Heads
See docs/therapy_system_implementation_plan.md for implementation details.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class GlobalConfidenceAggregator(nn.Module):
    """
    Global Confidence Aggregator (formerly UncertaintyHead).
    
    FIXED: System-level uncertainty aggregation, not isolated head.
    Architecturally declares that uncertainty suppresses action globally.
    
    Inputs (future expansion):
    - Scene embedding
    - Motion residual
    - OCR entropy
    - Audio entropy
    
    Outputs:
    - global_confidence [B, 1] (inverse of uncertainty, [0, 1])
    - uncertainty_score [B, 1] (1 - confidence, for backward compatibility)
    
    Loss:
    - NLL loss
    - ECE calibration loss
    """
    
    def __init__(self, scene_dim: int = 256, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.scene_dim = scene_dim
        
        # Uncertainty estimation network (with LayerNorm and dropout for efficiency)
        self.fc1 = nn.Linear(scene_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.norm2 = nn.LayerNorm(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
    
    def forward(
        self, 
        scene_embedding: torch.Tensor,
        motion_residual: Optional[torch.Tensor] = None,
        ocr_entropy: Optional[torch.Tensor] = None,
        audio_entropy: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multi-modal uncertainty aggregation.
        
        FIXED: Accepts multiple uncertainty signals (future expansion).
        Even if not all inputs are implemented yet, architecturally declaring it matters.
        
        Arguments:
            scene_embedding: Scene embedding [B, scene_dim]
            motion_residual: Optional motion residual [B, motion_dim] (future)
            ocr_entropy: Optional OCR entropy [B, 1] (future)
            audio_entropy: Optional audio entropy [B, 1] (future)
        
        Returns:
            Dictionary with:
                - 'global_confidence': [B, 1] - Confidence [0, 1] (higher = more confident)
                - 'uncertainty_score': [B, 1] - Uncertainty [0, 1] (for backward compatibility)
        """
        # Process scene embedding (primary signal)
        x = self.relu(self.norm1(self.fc1(scene_embedding)))
        x = self.dropout(x)
        
        # Future expansion: aggregate additional uncertainty signals
        # Currently only scene embedding is used
        if motion_residual is not None or ocr_entropy is not None or audio_entropy is not None:
            # Placeholder for future multi-modal aggregation
            pass
        
        x = self.relu(self.norm2(self.fc2(x)))
        x = self.dropout(x)
        confidence = self.sigmoid(self.fc3(x))
        uncertainty = 1.0 - confidence
        
        return {
            'global_confidence': confidence,
            'uncertainty_score': uncertainty  # Backward compatibility
        }


# Backward compatibility alias
UncertaintyHead = GlobalConfidenceAggregator

