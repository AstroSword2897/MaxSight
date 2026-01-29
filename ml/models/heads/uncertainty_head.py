"""
Global Confidence Aggregator for MaxSight 3.0 (v2)

Confidence is a system-level control signal.
High uncertainty suppresses verbosity, alerts, and action intensity.

Key improvements:
- Confidence acts as global gain
- Supports additive multi-modal evidence
- Logit-space aggregation (better calibration)
- Explicit monotonicity (more uncertainty → less action)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class GlobalConfidenceAggregator(nn.Module):
    """
    Global Confidence Aggregator (v2).
    
    Confidence is a system-level control signal.
    High uncertainty suppresses verbosity, alerts, and action intensity.
    
    Inputs:
    - Scene embedding (primary)
    - Motion residual (optional)
    - OCR entropy (optional)
    - Audio entropy (optional)
    
    Outputs:
    - global_confidence [B, 1] (inverse of uncertainty, [0, 1])
    - confidence_logit [B, 1] (logit space, for training)
    - uncertainty_score [B, 1] (1 - confidence, for backward compatibility)
    """
    
    def __init__(
        self,
        scene_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.backbone = nn.Sequential(
            nn.Linear(scene_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Logit space (NOT sigmoid yet)
        self.confidence_logit = nn.Linear(hidden_dim // 2, 1)
    
    def forward(
        self,
        scene_embedding: torch.Tensor,
        motion_residual: Optional[torch.Tensor] = None,
        ocr_entropy: Optional[torch.Tensor] = None,
        audio_entropy: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multi-modal uncertainty aggregation.
        
        Args:
            scene_embedding: Scene embedding [B, scene_dim]
            motion_residual: Optional motion residual [B, motion_dim] (future)
            ocr_entropy: Optional OCR entropy [B, 1] (future)
            audio_entropy: Optional audio entropy [B, 1] (future)
        
        Returns:
            Dictionary with:
                - 'global_confidence': [B, 1] - Confidence [0, 1]
                - 'confidence_logit': [B, 1] - Logit space (for training)
                - 'uncertainty_score': [B, 1] - Uncertainty [0, 1]
        """
        x = self.backbone(scene_embedding)
        
        # -------------------------------------------------
        # Future: additive uncertainty penalties
        # -------------------------------------------------
        if ocr_entropy is not None:
            x = x - ocr_entropy
        if audio_entropy is not None:
            x = x - audio_entropy
        
        confidence_logit = self.confidence_logit(x)
        confidence = torch.sigmoid(confidence_logit)
        uncertainty = 1.0 - confidence
        
        return {
            "global_confidence": confidence,
            "confidence_logit": confidence_logit,
            "uncertainty_score": uncertainty,  # Backward compatibility
        }


# Backward compatibility alias
UncertaintyHead = GlobalConfidenceAggregator
