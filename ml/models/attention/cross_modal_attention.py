"""
Cross-Modal Attention for MaxSight 3.0

Enables attention between vision, audio, and haptic modalities.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention module.
    
    Enables vision ↔ audio ↔ haptics attention for multimodal fusion.
    """
    
    def __init__(
        self,
        vision_dim: int,
        audio_dim: int,
        haptic_dim: int = 0,  # Optional
        embed_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Projection layers to common dimension
        self.vision_proj = nn.Linear(vision_dim, embed_dim)
        self.audio_proj = nn.Linear(audio_dim, embed_dim)
        
        if haptic_dim > 0:
            self.haptic_proj = nn.Linear(haptic_dim, embed_dim)
        else:
            self.haptic_proj = None
        
        # Cross-attention: Vision ↔ Audio
        self.vision_audio_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.audio_vision_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Layer norms
        self.norm_vision = nn.LayerNorm(embed_dim)
        self.norm_audio = nn.LayerNorm(embed_dim)
        
        # Output projection
        if haptic_dim > 0:
            self.output_proj = nn.Linear(embed_dim * 3, embed_dim)
        else:
            self.output_proj = nn.Linear(embed_dim * 2, embed_dim)
    
    def forward(
        self,
        vision_features: torch.Tensor,  # [B, N_vision, vision_dim]
        audio_features: torch.Tensor,   # [B, N_audio, audio_dim]
        haptic_features: Optional[torch.Tensor] = None  # [B, haptic_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through cross-modal attention.
        
        Args:
            vision_features: Vision features [B, N_vision, vision_dim]
            audio_features: Audio features [B, N_audio, audio_dim]
            haptic_features: Optional haptic features [B, haptic_dim]
        
        Returns:
            fused: Fused multimodal features [B, embed_dim]
            vision_enhanced: Enhanced vision features [B, N_vision, embed_dim]
            audio_enhanced: Enhanced audio features [B, N_audio, embed_dim]
        """
        B = vision_features.shape[0]
        
        # Project to common dimension
        vision_proj = self.vision_proj(vision_features)  # [B, N_vision, embed_dim]
        audio_proj = self.audio_proj(audio_features)    # [B, N_audio, embed_dim]
        
        # Cross-attention: Vision attends to Audio
        vision_enhanced, _ = self.vision_audio_attn(
            query=vision_proj,
            key=audio_proj,
            value=audio_proj
        )  # [B, N_vision, embed_dim]
        vision_enhanced = self.norm_vision(vision_proj + vision_enhanced)
        
        # Cross-attention: Audio attends to Vision
        audio_enhanced, _ = self.audio_vision_attn(
            query=audio_proj,
            key=vision_proj,
            value=vision_proj
        )  # [B, N_audio, embed_dim]
        audio_enhanced = self.norm_audio(audio_proj + audio_enhanced)
        
        # Global pooling and fusion
        vision_global = vision_enhanced.mean(dim=1)  # [B, embed_dim]
        audio_global = audio_enhanced.mean(dim=1)    # [B, embed_dim]
        
        # Optional: Incorporate haptic
        if haptic_features is not None and self.haptic_proj is not None:
            haptic_proj = self.haptic_proj(haptic_features)  # [B, embed_dim]
            fused = torch.cat([vision_global, audio_global, haptic_proj], dim=1)
            fused = self.output_proj(fused)  # [B, embed_dim]
        else:
            fused = torch.cat([vision_global, audio_global], dim=1)
            fused = self.output_proj(fused)  # [B, embed_dim]
        
        return fused, vision_enhanced, audio_enhanced


