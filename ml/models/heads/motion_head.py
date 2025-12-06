"""
Motion/Flow Head for MaxSight Therapy System

Outputs optical flow for motion tracking therapy tasks and temporal understanding.

PROJECT PHILOSOPHY & APPROACH:
=============================
This module implements optical flow estimation as part of MaxSight's therapy system.
Motion tracking is critical for understanding dynamic scenes and supporting motion-based
therapy exercises that help users develop visual tracking skills.

WHY MOTION TRACKING MATTERS:
---------------------------
Motion tracking enables:

1. Motion-based therapy: Exercises that train users to track moving objects
2. Dynamic scene understanding: Understanding how objects move in the environment
3. Temporal consistency: Better object tracking across video frames
4. Navigation assistance: Detecting moving obstacles (vehicles, people)

HOW IT CONNECTS TO THE PROBLEM STATEMENT:
------------------------------------------
The problem emphasizes "Skill Development Across Senses" - motion tracking therapy
exercises help users develop visual tracking skills, improving their ability to
navigate dynamic environments and track moving objects.

RELATIONSHIP TO BARRIER REMOVAL METHODS:
----------------------------------------
1. SKILL DEVELOPMENT: Enables motion tracking therapy exercises
2. ENVIRONMENTAL STRUCTURING: Provides motion information for dynamic scene understanding
3. NAVIGATION ASSISTANCE: Detects moving obstacles for safer navigation

TECHNICAL DESIGN DECISIONS:
---------------------------
- Coarse-to-fine architecture: Initial coarse prediction + optional refinement
- BatchNorm: Training stability for convolutional layers
- Smoothness regularization: Encourages smooth flow fields (more realistic)
- Tanh output: Normalizes flow to [-1, 1] range for interpretability

Phase 2: Therapy Heads
See docs/therapy_system_implementation_plan.md for implementation details.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Union


class MotionHead(nn.Module):
    """
    Motion/flow head for therapy tasks and temporal understanding.
    
    WHY THIS CLASS EXISTS:
    ----------------------
    Optical flow estimation enables motion tracking therapy exercises and dynamic scene
    understanding. This head processes temporal features to estimate per-pixel motion
    vectors (u, v) that represent how pixels move between frames.
    
    This information enables:
    - Motion tracking therapy: Exercises that train users to track moving objects
    - Dynamic scene understanding: Understanding object movement patterns
    - Temporal consistency: Better object tracking across video frames
    
    Architecture:
    - Input: Temporal features [B, C, H, W] from temporal encoder
    - Coarse network: Initial motion estimation
    - Refinement network (optional): Refines coarse prediction for better accuracy
    - Output: Motion flow [B, 2, H, W] with (u, v) channels in [-1, 1] range
    
    Arguments:
        in_channels: Number of input channels from temporal encoder (default: 128)
        hidden_channels: Hidden layer channels (default: 64)
        use_refinement: Enable refinement module for better accuracy (default: True)
    """
    
    def __init__(
        self, 
        in_channels: int = 128,
        hidden_channels: int = 64,
        use_refinement: bool = True
    ):
        """
        Initialize motion head.
        
        Arguments:
            in_channels: Number of input channels from temporal encoder
            hidden_channels: Hidden layer channels for feature extraction
            use_refinement: Enable refinement module for better flow accuracy
        """
        super().__init__()
        self.in_channels = in_channels
        self.use_refinement = use_refinement
        
        # Coarse motion estimation
        # WHY COARSE-TO-FINE:
        # - Initial coarse prediction captures large motions
        # - Refinement module corrects fine details
        # - More accurate than single-stage prediction
        self.coarse_net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),  # Training stability
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True)
        )
        
        # Initial flow prediction
        self.flow_pred = nn.Conv2d(hidden_channels // 2, 2, kernel_size=1)  # (u, v) channels
        
        # Optional refinement module for better accuracy
        if use_refinement:
            # WHY REFINEMENT:
            # - Corrects errors in coarse prediction
            # - Handles fine details and small motions
            # - Residual connection preserves coarse prediction
            self.refinement_net = nn.Sequential(
                nn.Conv2d(hidden_channels // 2 + 2, hidden_channels // 2, 
                         kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_channels // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels // 2, 2, kernel_size=1)
            )
        
        self.tanh = nn.Tanh()  # Normalize to [-1, 1] range
    
    def forward(
        self, 
        temporal_features: torch.Tensor,
        return_features: bool = False
    ) -> Union[torch.Tensor, Dict[str, Union[torch.Tensor, None]]]:
        """
        Forward pass to generate motion flow.
        
        CRITICAL INPUT REQUIREMENTS:
        ----------------------------
        - Input must be temporal features from temporal encoder
        - Input shape must be [B, C, H, W] where C matches in_channels
        - Features should be normalized (handled by temporal encoder)
        
        Arguments:
            temporal_features: Temporal features [B, C, H, W]
            return_features: If True, return intermediate features for loss computation
        
        Returns:
            Motion flow [B, 2, H, W] - (u, v) channels, normalized to [-1, 1]
            If return_features=True, returns dict with 'flow', 'features', and 'coarse_flow'
        """
        # Validate input
        if temporal_features.dim() != 4:
            raise ValueError(f"Expected 4D tensor [B, C, H, W], got {temporal_features.shape}")
        
        B, C, H, W = temporal_features.shape
        if C != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channels, got {C}. "
                f"Ensure input features match head configuration."
            )
        
        # Extract features
        features = self.coarse_net(temporal_features)
        
        # Coarse flow prediction
        coarse_flow = self.flow_pred(features)
        
        # Optional refinement
        if self.use_refinement:
            # Concatenate features and coarse flow for refinement
            refinement_input = torch.cat([features, coarse_flow], dim=1)
            flow_residual = self.refinement_net(refinement_input)
            # Residual connection: coarse + refinement
            motion = self.tanh(coarse_flow + flow_residual)
        else:
            motion = self.tanh(coarse_flow)
        
        # Validate output
        if torch.isnan(motion).any() or torch.isinf(motion).any():
            raise RuntimeError(
                "NaN/Inf detected in motion flow. Check input features and model initialization."
            )
        
        # Compute motion magnitude (speed)
        flow_magnitude = torch.sqrt(motion[:, 0]**2 + motion[:, 1]**2)  # [B, H, W]
        
        if return_features:
            result: Dict[str, Union[torch.Tensor, None]] = {
                'flow': motion,
                'flow_magnitude': flow_magnitude,
                'features': features,
            }
            if self.use_refinement:
                result['coarse_flow'] = coarse_flow
            else:
                result['coarse_flow'] = None
            return result
        
        return motion
    
    def compute_smoothness_loss(
        self, 
        flow: torch.Tensor, 
        image: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute edge-aware smoothness regularization loss.
        
        WHY EDGE-AWARE SMOOTHNESS:
        --------------------------
        Real optical flow fields are smooth except at motion boundaries (edges).
        Edge-aware smoothness loss allows discontinuities at image edges, leading to
        more accurate flow predictions that respect object boundaries.
        
        Arguments:
            flow: Predicted flow [B, 2, H, W]
            image: Optional input image [B, C, H, W] for edge detection
        
        Returns:
            Smoothness loss scalar
        """
        # Compute flow gradients
        flow_grad_x = torch.abs(flow[:, :, :, :-1] - flow[:, :, :, 1:])  # [B, 2, H, W-1]
        flow_grad_y = torch.abs(flow[:, :, :-1, :] - flow[:, :, 1:, :])  # [B, 2, H-1, W]
        
        if image is not None:
            # Compute image gradients for edge detection
            # Convert to grayscale if multi-channel
            if image.shape[1] == 3:
                gray = image.mean(dim=1, keepdim=True)  # [B, 1, H, W]
            else:
                gray = image
            
            # Image gradients
            img_grad_x = torch.abs(gray[:, :, :, :-1] - gray[:, :, :, 1:])  # [B, 1, H, W-1]
            img_grad_y = torch.abs(gray[:, :, :-1, :] - gray[:, :, 1:, :])  # [B, 1, H-1, W]
            
            # Weight flow gradients inversely by image gradients
            # Less smoothness penalty at edges (where image gradients are high)
            weight_x = torch.exp(-img_grad_x.mean(dim=1, keepdim=True))  # [B, 1, H, W-1]
            weight_y = torch.exp(-img_grad_y.mean(dim=1, keepdim=True))  # [B, 1, H-1, W]
            
            # Expand weights to match flow gradient dimensions
            weight_x = weight_x.expand_as(flow_grad_x)
            weight_y = weight_y.expand_as(flow_grad_y)
            
            # Weighted smoothness
            smoothness = (
                (flow_grad_x * weight_x).mean() + 
                (flow_grad_y * weight_y).mean()
            )
        else:
            # Standard smoothness (no edge awareness)
            smoothness = (flow_grad_x.mean() + flow_grad_y.mean())
        
        return smoothness
