"""
Depth/Focus Head

Outputs depth map and near/mid/far classification.

IMPORTANT DESIGN DECISIONS:
- Depth is normalized [0, 1], not metric. This is a modeling assumption.
  If you need metric depth, scale in post-processing or use softplus activation.
- Uncertainty is used in loss to weight depth errors (uncertainty-weighted loss).
- Zone classification uses depth features for consistency.
- GroupNorm instead of BatchNorm for robustness to small batches.
- Separate branches for depth and uncertainty (they need different signals).
- Multi-scale features from FPN can be passed for better depth estimation.

Phase 2: Therapy Heads
See docs/therapy_system_implementation_plan.md for implementation details.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class DepthHead(nn.Module):
    """
    Depth/focus head for therapy tasks.
    
    DESIGN PHILOSOPHY:
    - Depth and uncertainty are separate but related tasks
    - Zone classification should be grounded in depth (not independent)
    - Multi-scale features improve depth smoothness
    - Normalized depth [0, 1] is a modeling assumption (documented)
    
    Inputs: fused FPN + temporal features [B, C, H, W]
    Optional: multi-scale FPN features for skip connections
    
    Output:
    - depth_map [B, H, W] - Normalized depth [0, 1] (NOT metric)
    - uncertainty [B, H, W] - Depth uncertainty [0, 1]
    - zones [B, 3] - Zone logits (near/mid/far) derived from depth
    
    Losses:
    - Uncertainty-weighted photometric loss
    - Zone classification loss (grounded in depth)
    """
    
    def __init__(
        self, 
        in_channels: int = 256 + 128,  # FPN + temporal
        dropout: float = 0.1,
        use_multi_scale: bool = True,
        depth_activation: str = 'sigmoid'  # 'sigmoid', 'softplus', or 'none'
    ):
        super().__init__()
        self.in_channels = in_channels
        self.use_multi_scale = use_multi_scale
        self.depth_activation = depth_activation
        
        # Separate branches for depth and uncertainty (they need different signals)
        # Depth branch: confident structure, clear gradients
        self.depth_branch = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 128),  # More robust than BatchNorm for small batches
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True)
        )
        
        # Uncertainty branch: ambiguity, texturelessness, motion
        self.uncertainty_branch = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True)
        )
        
        # Depth map output (normalized [0, 1])
        if depth_activation == 'sigmoid':
            self.depth_conv = nn.Sequential(
                nn.Conv2d(64, 1, kernel_size=1),
                nn.Sigmoid()  # Normalized depth [0, 1] - NOT metric
            )
        elif depth_activation == 'softplus':
            self.depth_conv = nn.Sequential(
                nn.Conv2d(64, 1, kernel_size=1),
                nn.Softplus()  # Positive depth, avoids saturation
            )
        else:  # 'none'
            self.depth_conv = nn.Conv2d(64, 1, kernel_size=1)  # Raw depth (scale in loss)
        
        # Uncertainty output [0, 1]
        self.uncertainty_conv = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()  # Uncertainty in [0, 1]
        )
        
        # Multi-scale skip connection (if FPN features available)
        if use_multi_scale:
            # Project lower-resolution FPN features to match current resolution
            self.fpn_proj = nn.ModuleDict({
                'p3': nn.Conv2d(256, 64, kernel_size=1),  # Coarser scale
                'p4': nn.Conv2d(256, 64, kernel_size=1)   # Medium scale
            })
        
        # Zone classification: grounded in depth features
        # Uses depth statistics + pooled features for consistency
        # Constants for maintainability
        ZONE_DEPTH_FEAT_DIM = 64  # Depth feature channels
        ZONE_DEPTH_STATS_DIM = 3  # Percentiles (p25, p50, p75)
        ZONE_INPUT_DIM = ZONE_DEPTH_FEAT_DIM + ZONE_DEPTH_STATS_DIM  # 67
        
        self.zone_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ZONE_INPUT_DIM, 32),  # 64 + 3 = 67
            nn.LayerNorm(32),  # Better than BatchNorm for 1D features
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 3)  # near, mid, far - raw logits
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(
        self, 
        features: torch.Tensor,
        fpn_features: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Arguments:
            features: Fused FPN + temporal features [B, C, H, W]
            fpn_features: Optional dict with 'p3', 'p4' for multi-scale skip connections
        
        Returns:
            Dictionary with:
                - 'depth_map': [B, H, W] - Normalized depth [0, 1] (NOT metric)
                - 'uncertainty': [B, H, W] - Depth uncertainty [0, 1]
                - 'zones': [B, 3] - Zone logits (not softmaxed)
        """
        B, C, H, W = features.shape
        
        # Separate branches for depth and uncertainty
        depth_feat = self.depth_branch(features)  # [B, 64, H, W]
        uncertainty_feat = self.uncertainty_branch(features)  # [B, 32, H, W]
        
        # Multi-scale skip connections (if available)
        if self.use_multi_scale and fpn_features is not None:
            depth_feat_list = [depth_feat]
            
            # Add coarser scales (upsampled to match resolution)
            for scale_name, proj in self.fpn_proj.items():
                if scale_name in fpn_features:
                    fpn_feat = fpn_features[scale_name]  # [B, 256, H_scale, W_scale]
                    proj_feat = proj(fpn_feat)  # [B, 64, H_scale, W_scale]
                    # Upsample to match current resolution
                    if proj_feat.shape[2:] != (H, W):
                        proj_feat = F.interpolate(
                            proj_feat, size=(H, W), mode='bilinear', align_corners=False
                        )
                    depth_feat_list.append(proj_feat)
            
            # Combine multi-scale features
            if len(depth_feat_list) > 1:
                depth_feat = torch.stack(depth_feat_list, dim=0).mean(dim=0)  # Average across scales
        
        # Depth map (normalized [0, 1] - NOT metric)
        depth_map = self.depth_conv(depth_feat)
        if depth_map.shape[1] == 1:
            depth_map = depth_map.view(B, H, W)
        else:
            depth_map = depth_map.squeeze(1)  # [B, H, W]
        
        # Uncertainty
        uncertainty = self.uncertainty_conv(uncertainty_feat)
        if uncertainty.shape[1] == 1:
            uncertainty = uncertainty.view(B, H, W)
        else:
            uncertainty = uncertainty.squeeze(1)  # [B, H, W]
        
        # Zone classification: grounded in depth with distributional statistics
        # Zones are distributional, not scalar - use percentiles, not just mean
        depth_flat = depth_map.view(B, -1)  # [B, H*W]
        
        # Compute depth percentiles (p25, p50, p75) for distributional awareness
        p25 = torch.quantile(depth_flat, 0.25, dim=1)  # [B]
        p50 = torch.quantile(depth_flat, 0.50, dim=1)  # [B] (median)
        p75 = torch.quantile(depth_flat, 0.75, dim=1)  # [B]
        depth_stats = torch.stack([p25, p50, p75], dim=1)  # [B, 3]
        
        # Pool depth features and concatenate with depth statistics
        depth_pooled = F.adaptive_avg_pool2d(depth_feat, 1).view(B, -1)  # [B, 64]
        zone_input = torch.cat([depth_pooled, depth_stats], dim=1)  # [B, 67] (64 + 3)
        
        zones = self.zone_head(zone_input)  # [B, 3] - raw logits
        
        return {
            'depth_map': depth_map,  # Normalized [0, 1], NOT metric
            'uncertainty': uncertainty,  # [0, 1]
            'zones': zones  # Raw logits
        }
