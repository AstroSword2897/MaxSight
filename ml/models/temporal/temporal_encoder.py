"""
Temporal Encoder Module for MaxSight 3.0

Enhanced with ConvLSTM and TimeSformer for advanced temporal processing.
Handles temporal processing of video sequences:
- Motion features (ConvLSTM)
- Long-range temporal dependencies (TimeSformer)
- Temporal consistency
- Flicker detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
from .conv_lstm import ConvLSTM
from .temporal_transformer import TimeSformer


class TemporalEncoder(nn.Module):
    """
    Enhanced temporal encoder for video sequence processing.
    
    Uses ConvLSTM for motion tracking and TimeSformer for long-range dependencies.
    Outputs motion features, temporal consistency, and flicker detection.
    
    Architecture:
    - ConvLSTM: Motion tracking for people, vehicles, obstacles
    - TimeSformer: Long-range temporal dependencies
    - Output: motion features, temporal consistency, flicker detection
    
    Input: [B, C, T, H, W] or [B, T, C, H, W] - Batch of video frames
    Output: Dict with motion features, consistency, flicker, temporal_context
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        num_frames: int = 8,
        hidden_dim: int = 256,
        vit_embed_dim: int = 768,
        use_conv_lstm: bool = True,
        use_timesformer: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_frames = num_frames
        self.hidden_dim = hidden_dim
        self.use_conv_lstm = use_conv_lstm
        self.use_timesformer = use_timesformer
        
        # ConvLSTM for motion tracking
        if use_conv_lstm:
            self.conv_lstm = ConvLSTM(
                input_dim=in_channels,
                hidden_dim=hidden_dim,
                kernel_size=3,
                num_layers=2
            )
        
        # TimeSformer for long-range temporal dependencies
        if use_timesformer:
            self.timesformer = TimeSformer(
                embed_dim=vit_embed_dim,
                num_heads=12,
                num_layers=12,
                num_frames=num_frames
            )
        
        # Motion feature head (from ConvLSTM output)
        if use_conv_lstm:
            self.motion_head = nn.Sequential(
                nn.Conv2d(hidden_dim, 2, kernel_size=1),  # u, v motion
                nn.Tanh()  # Normalize to [-1, 1]
            )
        
        # Temporal consistency head
        self.consistency_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Linear(hidden_dim if use_conv_lstm else in_channels, 1),
            nn.Sigmoid()
        )
        
        # Flicker detection head
        self.flicker_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Linear(hidden_dim if use_conv_lstm else in_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        frames: torch.Tensor,
        vit_patch_tokens: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through enhanced temporal encoder.
        
        Arguments:
            frames: Video frames [B, C, T, H, W] or [B, T, C, H, W]
            vit_patch_tokens: Optional ViT patch tokens [B, T, N_patches, embed_dim]
        
        Returns:
            Dictionary with:
                - 'motion': [B, 2, H, W] - Motion flow (u, v)
                - 'consistency': [B, 1] - Temporal consistency score
                - 'flicker': [B, 1] - Flicker detection score
                - 'temporal_context': [B, embed_dim] - Long-range temporal context (if TimeSformer used)
        """
        B = frames.shape[0]
        
        # Handle different input formats
        if frames.dim() == 5:
            if frames.shape[1] == self.in_channels:
                # [B, C, T, H, W]
                frames_seq = frames.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
            else:
                # [B, T, C, H, W]
                frames_seq = frames
        else:
            raise ValueError(f"Expected 5D input, got {frames.dim()}D")
        
        H, W = frames_seq.shape[-2], frames_seq.shape[-1]
        
        outputs = {}
        
        # ConvLSTM for motion tracking
        if self.use_conv_lstm:
            # Extract features from each frame (simplified - in practice would use CNN features)
            # For now, assume frames_seq is already feature maps
            motion_features, (h, c) = self.conv_lstm(frames_seq)  # [B, T, hidden_dim, H, W]
            
            # Use last frame's motion features
            motion_last = motion_features[:, -1]  # [B, hidden_dim, H, W]
            
            # Motion flow prediction
            motion = self.motion_head(motion_last)  # [B, 2, H, W]
            outputs['motion'] = motion
            
            # Temporal consistency from motion features
            consistency_feat = motion_last
        else:
            # Fallback: use last frame
            consistency_feat = frames_seq[:, -1]  # [B, C, H, W]
            outputs['motion'] = torch.zeros(B, 2, H, W, device=frames.device)
        
        # Temporal consistency score
        consistency = self.consistency_head(consistency_feat).squeeze(-1).squeeze(-1)  # [B, 1]
        outputs['consistency'] = consistency.unsqueeze(1) if consistency.dim() == 1 else consistency
        
        # Flicker detection
        flicker = self.flicker_head(consistency_feat).squeeze(-1).squeeze(-1)  # [B, 1]
        outputs['flicker'] = flicker.unsqueeze(1) if flicker.dim() == 1 else flicker
        
        # TimeSformer for long-range temporal context
        if self.use_timesformer and vit_patch_tokens is not None:
            temporal_context = self.timesformer(vit_patch_tokens)  # [B, embed_dim]
            outputs['temporal_context'] = temporal_context
        
        return outputs


class TemporalBuffer:
    """
    Buffer for maintaining temporal context across frames.
    
    Maintains a sliding window of recent frames for temporal processing.
    """
    
    def __init__(self, buffer_size: int = 5):
        self.buffer_size = buffer_size
        self.buffer = []
    
    def add_frame(self, frame: torch.Tensor):
        """Add a new frame to the buffer."""
        self.buffer.append(frame)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
    
    def get_sequence(self) -> Optional[torch.Tensor]:
        """
        Get the current sequence of frames.
        
        Returns:
            Tensor [T, C, H, W] if buffer is full, None otherwise
        """
        if len(self.buffer) < self.buffer_size:
            return None
        return torch.stack(self.buffer, dim=0)
    
    def clear(self):
        """Clear the buffer."""
        self.buffer = []

