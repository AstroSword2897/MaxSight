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
            pred_contrast: Predicted contrast map [B, H, W] or [B, C, H, W]
            target_contrast: Ground truth contrast map [B, H, W] or [B, C, H, W]
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
            # Ensure target has channel dimension for edge computation
            if target_contrast.dim() == 3:
                target_4d = target_contrast.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
            else:
                target_4d = target_contrast
            
            # Compute edge map from target using Sobel filters
            edge_map = self._compute_edge_map(target_4d)
            
            # Ensure pred has same dimensions as target_4d
            if pred_contrast.dim() == 3:
                pred_4d = pred_contrast.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
            else:
                pred_4d = pred_contrast
            
            # Weight the pixel-wise loss by edge strength
            pixel_wise_loss = torch.abs(pred_4d - target_4d)
            edge_weighted_loss = pixel_wise_loss * (1.0 + edge_map)
            edge_aware_loss = edge_weighted_loss.mean()
            
            losses['edge_aware_loss'] = edge_aware_loss
            losses['total_loss'] = 0.5 * l1_loss + 0.5 * edge_aware_loss
        else:
            losses['total_loss'] = l1_loss
        
        return losses
    
    def _compute_edge_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute edge map using Sobel filters.
        
        WHY THIS METHOD:
        ----------------
        Edge maps are computed directly from the input tensor using Sobel filters, which detect
        gradients (edges) in the image. This is more flexible than using pre-computed edge maps
        and works with any input tensor shape.
        
        Arguments:
            x: Input tensor of shape [B, C, H, W] or [B, H, W] or [H, W]
        
        Returns:
            Edge magnitude map of shape [B, 1, H, W] or [1, H, W]
        """
        # Ensure 4D tensor - handle all input shapes
        original_dim = x.dim()
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)  # [H, W] -> [1, 1, H, W]
            squeeze_output = True
        elif x.dim() == 3:
            x = x.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
            squeeze_output = False
        else:
            squeeze_output = False
        
        # Convert to grayscale if multichannel (vectorized)
        if x.shape[1] > 1:
            # Use standard RGB to grayscale weights (efficient broadcasting)
            gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        else:
            gray = x
        
        # Apply Sobel filters with padding (buffers already on correct device)
        # Cast sobel filters to match input dtype for efficiency
        # Type assertion: buffers are guaranteed to be tensors
        sobel_x_tensor: torch.Tensor = self.sobel_x.to(dtype=x.dtype)  # type: ignore
        sobel_y_tensor: torch.Tensor = self.sobel_y.to(dtype=x.dtype)  # type: ignore
        grad_x = F.conv2d(gray, sobel_x_tensor, padding=1)
        grad_y = F.conv2d(gray, sobel_y_tensor, padding=1)
        
        # Compute gradient magnitude (vectorized)
        edge_map = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        
        # Normalize to [0, 1] range (efficient: single min/max call per batch)
        # edge_map is guaranteed to be [B, 1, H, W] at this point (4D)
        if edge_map.dim() == 4:
            B, C, H, W = edge_map.shape
            edge_flat = edge_map.view(B, -1)  # [B, H*W]
            edge_min = edge_flat.min(dim=1, keepdim=True)[0]  # [B, 1]
            edge_max = edge_flat.max(dim=1, keepdim=True)[0]  # [B, 1]
            
            # Reshape for broadcasting: [B, 1] -> [B, 1, 1, 1] to match [B, 1, H, W]
            edge_min = edge_min.view(B, 1, 1, 1)
            edge_max = edge_max.view(B, 1, 1, 1)
            
            # Avoid division by zero with efficient masking
            range_mask = (edge_max > edge_min).float()
            edge_map = range_mask * (edge_map - edge_min) / (edge_max - edge_min + 1e-8) + (1 - range_mask) * torch.zeros_like(edge_map)
        else:
            # Fallback for unexpected shapes (shouldn't happen, but safe)
            edge_max_val = edge_map.max()
            if edge_max_val > 0:
                edge_map = edge_map / (edge_max_val + 1e-8)
        
        # Restore original dimensionality if needed
        if original_dim == 2:
            edge_map = edge_map.squeeze(0).squeeze(0)  # [1, 1, H, W] -> [H, W]
        elif original_dim == 3:
            edge_map = edge_map.squeeze(1)  # [B, 1, H, W] -> [B, H, W]
        
        return edge_map

