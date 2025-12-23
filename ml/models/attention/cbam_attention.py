"""
Attention Modules for MaxSight 3.0

Includes CBAM (Convolutional Block Attention Module) and SE (Squeeze-and-Excitation) blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ChannelAttention(nn.Module):
    """
    Channel Attention Module (CAM).
    
    Uses both average and max pooling to capture channel-wise dependencies.
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP for both pooling operations
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply channel attention.
        
        Args:
            x: Input features [B, C, H, W]
        
        Returns:
            Attention-weighted features [B, C, H, W]
        """
        # Average pooling branch
        avg_out = self.shared_mlp(self.avg_pool(x))  # [B, C, 1, 1]
        
        # Max pooling branch
        max_out = self.shared_mlp(self.max_pool(x))  # [B, C, 1, 1]
        
        # Combine and apply sigmoid
        channel_weights = self.sigmoid(avg_out + max_out)  # [B, C, 1, 1]
        
        # Scale input features
        return x * channel_weights


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (SAM).
    
    Focuses on 'where' to pay attention spatially.
    """
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        
        # Convolution to generate spatial attention map
        self.conv = nn.Conv2d(
            2, 1, kernel_size, padding=kernel_size // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial attention.
        
        Args:
            x: Input features [B, C, H, W]
        
        Returns:
            Attention-weighted features [B, C, H, W]
        """
        # Channel-wise average pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        
        # Channel-wise max pooling
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        
        # Concatenate and apply convolution
        spatial_features = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        spatial_weights = self.sigmoid(self.conv(spatial_features))  # [B, 1, H, W]
        
        # Scale input features
        return x * spatial_weights


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    
    Combines channel attention and spatial attention sequentially.
    """
    
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply CBAM attention.
        
        Args:
            x: Input features [B, C, H, W]
        
        Returns:
            Attention-weighted features [B, C, H, W]
        """
        # Apply channel attention first
        x = self.channel_attention(x)
        
        # Then apply spatial attention
        x = self.spatial_attention(x)
        
        return x


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block.
    
    Architecture:
    1. Squeeze: Global average pooling
    2. Excitation: FC layers with ReLU and Sigmoid
    3. Scale: Multiply with input features
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Excitation: FC layers
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SE attention.
        
        Args:
            x: Input features [B, C, H, W]
        
        Returns:
            Attention-weighted features [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # Squeeze: global average pooling
        y = self.avg_pool(x).reshape(B, C)  # [B, C]
        
        # Excitation: FC layers
        y = self.fc(y).reshape(B, C, 1, 1)  # [B, C, 1, 1]
        
        # Scale: multiply with input
        return x * y


