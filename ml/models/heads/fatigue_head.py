"""
Fatigue/Gaze Head for MaxSight Therapy System

Outputs fatigue score, blink rate, and fixation stability for adaptive assistance
and therapy task generation.

PROJECT PHILOSOPHY & APPROACH:
=============================
This module implements fatigue and gaze tracking as part of MaxSight's therapy system.
Understanding user state (fatigue, attention, cognitive load) enables the system to
adapt assistance levels appropriately - reducing detail when fatigued, increasing support
when attention is low, and adjusting therapy task difficulty based on user capabilities.

WHY FATIGUE TRACKING MATTERS:
-----------------------------
Fatigue and attention levels significantly impact how users interact with assistive
technology. This head enables:

1. Adaptive assistance: Adjust verbosity and detail based on fatigue levels
2. Therapy task adaptation: Modify task difficulty when user is fatigued
3. Safety monitoring: Detect when user needs rest or reduced cognitive load
4. Skill development: Track attention patterns to support vision therapy

HOW IT CONNECTS TO THE PROBLEM STATEMENT:
------------------------------------------
The problem emphasizes "Routine Workflow" and "Skill Development" - understanding user
state enables the system to adapt assistance appropriately, supporting both immediate
needs and long-term skill development. This head provides the user state information
needed for adaptive assistance.

RELATIONSHIP TO BARRIER REMOVAL METHODS:
----------------------------------------
1. SKILL DEVELOPMENT: Tracks attention and fatigue to adapt therapy exercises
2. ROUTINE WORKFLOW: Adapts to user patterns based on fatigue and attention levels
3. ADAPTIVE ASSISTANCE: Adjusts information density based on user state

TECHNICAL DESIGN DECISIONS:
---------------------------
- Shared backbone: Reduces parameters and encourages learning common features
- LayerNorm + Dropout: Better generalization and training stability
- Task-specific heads: Allows fine-tuning for each output while sharing features
- Multiple outputs: Fatigue, blink rate, and fixation stability provide comprehensive state

Phase 2: Therapy Heads
See docs/therapy_system_implementation_plan.md for implementation details.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class FatigueHead(nn.Module):
    """
    Fatigue/gaze head for therapy tasks and adaptive assistance.
    
    WHY THIS CLASS EXISTS:
    ----------------------
    Understanding user state (fatigue, attention, cognitive load) is critical for adaptive
    assistance. This head processes eye tracking and temporal features to estimate:
    - Fatigue levels: When user needs rest or reduced cognitive load
    - Blink rate: Indicator of attention and fatigue
    - Fixation stability: Measure of focus and visual attention quality
    
    This information enables the system to adapt assistance levels, adjust therapy task
    difficulty, and provide appropriate feedback based on user state.
    
    Architecture:
    - Input: Eye features [B, eye_dim] + Temporal features [B, temporal_dim]
    - Shared backbone: Extracts common features with regularization
    - Task-specific heads: Generate fatigue, blink rate, and fixation stability scores
    - Output: All scores in [0, 1] range for interpretability
    
    Arguments:
        eye_dim: Dimension of eye model features (default: 4)
        temporal_dim: Dimension of temporal features (default: 128)
        hidden_dim: Hidden layer dimension (default: 64)
        dropout: Dropout probability for regularization (default: 0.1)
    """
    
    def __init__(
        self, 
        eye_dim: int = 4, 
        temporal_dim: int = 128,
        hidden_dim: int = 64,
        dropout: float = 0.1
    ):
        """
        Initialize fatigue head.
        
        Arguments:
            eye_dim: Dimension of eye model features (blink_prob + fixation + pupil_size)
            temporal_dim: Dimension of temporal features from temporal encoder
            hidden_dim: Hidden layer dimension for shared backbone
            dropout: Dropout probability for regularization
        """
        super().__init__()
        self.eye_dim = eye_dim
        self.temporal_dim = temporal_dim
        self.hidden_dim = hidden_dim
        
        # Shared feature extraction with dropout for regularization
        # WHY SHARED BACKBONE:
        # - Reduces parameters (more efficient)
        # - Encourages learning common features across tasks
        # - Better generalization with shared representations
        self.shared_net = nn.Sequential(
            nn.Linear(eye_dim + temporal_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Better than BatchNorm for variable batch sizes
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),  # Regularization to prevent overfitting
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Task-specific heads with residual connections
        # WHY TASK-SPECIFIC HEADS:
        # - Allows fine-tuning for each output while sharing features
        # - More expressive than single shared head
        head_input_dim = hidden_dim // 2
        self.fatigue_head = self._make_head(head_input_dim)
        self.blink_rate_head = self._make_head(head_input_dim)
        self.fixation_stability_head = self._make_head(head_input_dim)
    
    def _make_head(self, input_dim: int) -> nn.Module:
        """
        Create a task-specific head with additional capacity.
        
        Arguments:
            input_dim: Input dimension for the head
        
        Returns:
            Sequential module for task-specific prediction
        """
        return nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Output in [0, 1] range
        )
    
    def forward(
        self,
        eye_features: torch.Tensor,
        temporal_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass to generate fatigue and gaze predictions.
        
        CRITICAL INPUT REQUIREMENTS:
        ----------------------------
        - eye_features: Must be [B, eye_dim] from EyeModel
        - temporal_features: Must be [B, temporal_dim] from temporal encoder
        - Both inputs must be on same device and have same batch size
        
        Arguments:
            eye_features: Eye model features [B, eye_dim]
            temporal_features: Temporal features [B, temporal_dim]
        
        Returns:
            Dictionary with:
                - 'fatigue_score': [B, 1] - Fatigue level [0, 1] (0=alert, 1=fatigued)
                - 'blink_rate': [B, 1] - Blink rate [0, 1] (0=low, 1=high)
                - 'fixation_stability': [B, 1] - Fixation stability [0, 1] (0=unstable, 1=stable)
                - 'shared_features': [B, hidden_dim//2] - Shared features for analysis/visualization
        """
        # Validate inputs
        if eye_features.dim() != 2:
            raise ValueError(f"Expected 2D eye_features [B, eye_dim], got {eye_features.shape}")
        if temporal_features.dim() != 2:
            raise ValueError(f"Expected 2D temporal_features [B, temporal_dim], got {temporal_features.shape}")
        
        B_eye = eye_features.shape[0]
        B_temp = temporal_features.shape[0]
        if B_eye != B_temp:
            raise ValueError(f"Batch size mismatch: eye_features {B_eye} vs temporal_features {B_temp}")
        
        if eye_features.shape[1] != self.eye_dim:
            raise ValueError(f"Expected eye_dim={self.eye_dim}, got {eye_features.shape[1]}")
        if temporal_features.shape[1] != self.temporal_dim:
            raise ValueError(f"Expected temporal_dim={self.temporal_dim}, got {temporal_features.shape[1]}")
        
        # Combine and extract shared features
        combined = torch.cat([eye_features, temporal_features], dim=1)
        shared = self.shared_net(combined)
        
        # Validate shared features
        if torch.isnan(shared).any() or torch.isinf(shared).any():
            raise RuntimeError(
                "NaN/Inf detected in shared features. Check input features and model initialization."
            )
        
        # Generate predictions
        outputs = {
            'fatigue_score': self.fatigue_head(shared),
            'blink_rate': self.blink_rate_head(shared),
            'fixation_stability': self.fixation_stability_head(shared),
            'shared_features': shared  # Useful for analysis/visualization
        }
        
        # Validate outputs
        for key, value in outputs.items():
            if torch.isnan(value).any() or torch.isinf(value).any():
                raise RuntimeError(f"NaN/Inf detected in {key}. Check model initialization.")
        
        return outputs
