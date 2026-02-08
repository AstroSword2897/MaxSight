"""Audio Encoder for Multi-Vector Retrieval Encodes environmental audio using CNN + Transformer."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class AudioEncoder(nn.Module):
    """Audio encoder for retrieval. Architecture: - CNN + Transformer on spectrograms - Environmental sound embeddings - Spatial audio embeddings (direction + distance)"""
    
    def __init__(
        self,
        input_dim: int = 128,  # Spectrogram or MFCC features.
        embed_dim: int = 256,
        num_heads: int = 8
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        # Spectrogram CNN.
        self.spectrogram_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((None, embed_dim // 4))
        )
        
        # Transformer for temporal modeling.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Spatial audio head (direction + distance)
        self.spatial_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 6)
        )
    
    def forward(
        self,
        audio_features: torch.Tensor,  # [B, T, input_dim] or [B, input_dim].
        stereo_channels: Optional[torch.Tensor] = None  # [B, T, 2].
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode audio to embeddings."""
        B = audio_features.shape[0]
        
        # Handle different input formats.
        if audio_features.dim() == 2:
            audio_features = audio_features.unsqueeze(1)  # [B, 1, input_dim].
        
        # If spectrogram format [B, T, F].
        if audio_features.dim() == 3:
            T, F = audio_features.shape[1], audio_features.shape[2]
            spec = audio_features.unsqueeze(1)  # [B, 1, T, F].
            
            # Apply CNN.
            cnn_out = self.spectrogram_cnn(spec)  # [B, 64, T', embed_dim//4].
            cnn_out = cnn_out.permute(0, 2, 1, 3).contiguous()
            cnn_out = cnn_out.reshape(B, -1, self.embed_dim)
            
            # Apply transformer.
            audio_embed = self.transformer(cnn_out)  # [B, T', embed_dim].
            audio_embed = audio_embed.mean(dim=1)  # [B, embed_dim].
        else:
            # Simple projection.
            audio_embed = nn.Linear(audio_features.shape[-1], self.embed_dim).to(audio_features.device)(audio_features.mean(dim=1))
        
        # Spatial features.
        spatial_features = self.spatial_head(audio_embed)  # [B, 6].
        
        # L2 normalize audio embedding.
        audio_embed = F.normalize(audio_embed, p=2, dim=1)
        
        return audio_embed, spatial_features








