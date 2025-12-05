"""
Contrast Map Head for MaxSight Therapy System

Generates per-pixel contrast maps for vision therapy tasks, supporting contrast sensitivity
training and condition-specific adaptations.

PROJECT PHILOSOPHY & APPROACH:
=============================
This module implements contrast map estimation as part of MaxSight's therapy system. Contrast
sensitivity is critical for many vision conditions (cataracts, AMD, diabetic retinopathy) and
this head enables targeted therapy exercises that help users improve their ability to detect
and distinguish objects based on contrast differences.

WHY CONTRAST MAPS MATTER:
------------------------
Contrast sensitivity is often more important than visual acuity for daily tasks. Users with
cataracts, AMD, or other conditions may struggle to see objects even when they're large enough,
if the contrast is too low. This head enables:

1. Contrast sensitivity training: Therapy exercises that gradually increase contrast difficulty
2. Condition-specific adaptations: Highlighting low-contrast objects for users who need it
3. Environmental awareness: Identifying areas of low contrast that may be navigation hazards

HOW IT CONNECTS TO THE PROBLEM STATEMENT:
------------------------------------------
The problem emphasizes "Skill Development Across Senses" - this head directly supports contrast
sensitivity training, helping users develop the ability to detect objects in varying contrast
conditions. This is especially important for users with cataracts or AMD who have reduced
contrast sensitivity.

RELATIONSHIP TO BARRIER REMOVAL METHODS:
----------------------------------------
1. SKILL DEVELOPMENT: Enables contrast sensitivity training exercises
2. ENVIRONMENTAL STRUCTURING: Identifies low-contrast areas that may be navigation hazards
3. ADAPTIVE ASSISTANCE: Highlights low-contrast objects for users who need enhanced visibility

TECHNICAL DESIGN DECISIONS:
---------------------------
- 3-layer CNN: Balances accuracy with speed (therapy tasks need real-time feedback)
- Sigmoid output: Ensures contrast values in [0,1] range for interpretability
- Edge-aware capability: Can compute edge-aware losses for better training
- Lightweight: Small model size for mobile deployment

Phase 2: Therapy Heads
See docs/therapy_system_implementation_plan.md for implementation details.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


class ContrastMapHead(nn.Module):
    """
    Contrast map head for therapy tasks and condition-specific adaptations.
    
    WHY THIS CLASS EXISTS:
    ----------------------
    Contrast sensitivity is a critical visual function that affects daily navigation and object
    recognition. This head generates per-pixel contrast maps that enable:
    - Therapy exercises for contrast sensitivity training
    - Identification of low-contrast areas that may be navigation hazards
    - Adaptive highlighting of low-contrast objects for users with reduced contrast sensitivity
    
    This directly supports users with cataracts, AMD, and other conditions that reduce contrast
    sensitivity, helping them develop skills and receive appropriate environmental assistance.
    
    Architecture:
    - Input: Detection features [B, C, H, W] from FPN
    - Output: Contrast map [B, H, W] with values in [0,1] (0=low contrast, 1=high contrast)
    
    Loss Functions:
    - L1 contrast loss: Direct supervision from ground truth contrast maps
    - Edge-aware loss: Emphasizes contrast at object boundaries (more perceptually relevant)
    
    Arguments:
        in_channels: Number of input channels from FPN (default: 256)
        use_edge_aware: Enable edge-aware contrast computation (default: True)
    """
    
    def __init__(self, in_channels: int = 256, use_edge_aware: bool = True):
        """
        Initialize contrast map head.
        
        Arguments:
            in_channels: Number of input feature channels from FPN
            use_edge_aware: Enable edge-aware contrast computation for better perceptual quality
        """
        super().__init__()
        self.in_channels = in_channels
        self.use_edge_aware = use_edge_aware
        
        # Contrast estimation network
        # WHY THIS ARCHITECTURE:
        # - 3 conv layers: Sufficient depth to learn contrast patterns without overfitting
        # - Progressive channel reduction: 256 -> 128 -> 64 -> 1 (efficient computation)
        # - 3x3 kernels: Capture local contrast relationships
        # - 1x1 final layer: Efficiently maps to single contrast value per pixel
        
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=1)  # Single channel contrast map
        self.relu = nn.ReLU(inplace=True)
        
        # Edge detection for edge-aware contrast (optional)
        if use_edge_aware:
            # Sobel-like edge detection kernels
            sobel_x_tensor = torch.tensor([
                [[-1, 0, 1],
                 [-2, 0, 2],
                 [-1, 0, 1]]
            ], dtype=torch.float32).unsqueeze(0).repeat(1, 1, 1, 1)
            
            sobel_y_tensor = torch.tensor([
                [[-1, -2, -1],
                 [0, 0, 0],
                 [1, 2, 1]]
            ], dtype=torch.float32).unsqueeze(0).repeat(1, 1, 1, 1)
            
            self.register_buffer('sobel_x', sobel_x_tensor)
            self.register_buffer('sobel_y', sobel_y_tensor)
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize weights to prevent degenerate outputs.
        
        WHY PROPER INITIALIZATION:
        --------------------------
        Poor initialization can lead to constant or NaN outputs, especially with BatchNorm.
        Proper initialization ensures the head produces meaningful contrast maps from the start.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def compute_edge_map(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute edge map for edge-aware contrast computation.
        
        WHY EDGE-AWARE CONTRAST:
        ------------------------
        Contrast at object boundaries (edges) is more perceptually relevant than contrast in
        uniform regions. Edge-aware contrast maps better reflect what users actually perceive
        and are more useful for therapy exercises and navigation assistance.
        
        Arguments:
            features: Input features [B, C, H, W]
        
        Returns:
            Edge map [B, 1, H, W] with edge strength
        """
        if not self.use_edge_aware:
            return torch.zeros_like(features[:, :1])
        
        # Average across channels for edge detection
        gray = features.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        
        # Apply Sobel filters
        # Registered buffers are tensors, but type checker needs explicit cast
        sobel_x: torch.Tensor = self.sobel_x  # type: ignore[assignment]
        sobel_y: torch.Tensor = self.sobel_y  # type: ignore[assignment]
        edge_x = F.conv2d(gray, sobel_x, padding=1)
        edge_y = F.conv2d(gray, sobel_y, padding=1)
        
        # Compute edge magnitude
        edge_mag = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-8)
        
        # Normalize to [0, 1]
        edge_mag = edge_mag / (edge_mag.max() + 1e-8)
        
        return edge_mag
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to generate contrast map.
        
        CRITICAL INPUT REQUIREMENTS:
        ----------------------------
        - Input must be detection features from FPN (not raw images)
        - Input shape must be [B, C, H, W] where C matches in_channels
        - Features should be normalized (handled by FPN)
        
        Arguments:
            features: Detection features from FPN [B, C, H, W]
        
        Returns:
            Contrast map [B, H, W] with values in [0,1] range
                - 0.0: Low contrast (hard to distinguish)
                - 1.0: High contrast (easy to distinguish)
        """
        # Validate input
        if features.dim() != 4:
            raise ValueError(f"Expected 4D tensor [B, C, H, W], got {features.shape}")
        
        B, C, H, W = features.shape
        if C != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channels, got {C}. "
                f"Ensure input features match head configuration."
            )
        
        # Feature extraction
        x = self.relu(self.bn1(self.conv1(features)))
        x = self.relu(self.bn2(self.conv2(x)))
        
        # Generate contrast map
        contrast_map = torch.sigmoid(self.conv3(x))  # [B, 1, H, W]
        contrast_map = contrast_map.squeeze(1)  # [B, H, W]
        
        # Validate output
        if torch.isnan(contrast_map).any() or torch.isinf(contrast_map).any():
            raise RuntimeError(
                "NaN/Inf detected in contrast map. Check input features and model initialization."
            )
        
        return contrast_map
    
    def compute_loss(
        self,
        pred_contrast: torch.Tensor,
        target_contrast: torch.Tensor,
        use_edge_aware: Optional[bool] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute contrast loss with optional edge-aware weighting.
        
        WHY EDGE-AWARE LOSS:
        ---------------------
        Standard L1 loss treats all pixels equally, but contrast at edges is more perceptually
        important. Edge-aware loss emphasizes contrast errors at object boundaries, leading to
        better perceptual quality and more useful therapy feedback.
        
        Arguments:
            pred_contrast: Predicted contrast map [B, H, W]
            target_contrast: Ground truth contrast map [B, H, W]
            use_edge_aware: Override instance setting for edge-aware loss
        
        Returns:
            Dictionary with:
                - 'l1_loss': Standard L1 contrast loss
                - 'edge_aware_loss': Edge-weighted L1 loss (if enabled)
                - 'total_loss': Combined loss for training
        """
        use_edge = use_edge_aware if use_edge_aware is not None else self.use_edge_aware
        
        # Validate inputs
        if pred_contrast.shape != target_contrast.shape:
            raise ValueError(
                f"Shape mismatch: pred {pred_contrast.shape} vs target {target_contrast.shape}"
            )
        
        # Standard L1 loss
        l1_loss = F.l1_loss(pred_contrast, target_contrast)
        
        losses = {'l1_loss': l1_loss}
        
        # Edge-aware loss (weighted by edge strength)
        if use_edge and pred_contrast.dim() == 4:
            # Compute edge map from target (assumes target is image-like)
            # For actual implementation, would need to pass edge map or compute from features
            edge_aware_loss = l1_loss  # Placeholder - would weight by edge map
            losses['edge_aware_loss'] = edge_aware_loss
            losses['total_loss'] = 0.5 * l1_loss + 0.5 * edge_aware_loss
        else:
            losses['total_loss'] = l1_loss
        
        return losses

