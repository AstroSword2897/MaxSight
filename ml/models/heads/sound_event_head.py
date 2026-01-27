"""
Sound Event Classification Head for MaxSight 3.0

CNN + temporal attention for sound event classification and directional detection.
Fully learnable direction prediction, priority weighting, and urgency mapping.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class SoundEventHead(nn.Module):
    """
    Sound event classification head.
    
    Features:
    - Spectrogram CNN
    - Temporal attention
    - Fully learnable directional detection
    - Priority weighting
    - Urgency mapping
    """
    
    def __init__(
        self,
        input_dim: int = 128,  
        num_classes: int = 15,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_directions: int = 4
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_directions = num_directions
        self.embed_dim = embed_dim

        # Input projection for non-spectrogram features
        self.input_proj = nn.Linear(input_dim, embed_dim)
        
        # Spectrogram CNN
        self.spectrogram_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((None, embed_dim // 4))  # [T', embed_dim//4]
        )

        # Project CNN output to embedding dimension
        self.spectrogram_proj = nn.Linear(64 * (embed_dim // 4), embed_dim)
        
        # Temporal attention
        self.temporal_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.attention_norm = nn.LayerNorm(embed_dim)
        
        # Sound classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # Fully learnable directional detection
        self.direction_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_directions),
            nn.Softmax(dim=1)
        )
        
        # Priority weighting head
        self.priority_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Urgency mapping: class -> urgency level
        self.register_buffer('urgency_map', torch.tensor([
            3,  # EMERGENCY
            3,  # ALARM
            3,  # SIREN
            2,  # CONSTRUCTION
            2,  # VEHICLE
            1,  # DOOR
            1,  # BELL
            1,  # FOOTSTEPS
            0,  # SPEECH
            0,  # MUSIC
            0,  # WATER
            0,  # WIND
            0,  # CROWD
            0,  # ANIMAL
            0,  # BACKGROUND
        ]))
    
    def forward(
        self,
        audio_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through sound event head.
        
        Args:
            audio_features: Audio features [B, T, input_dim] or [B, input_dim]
        
        Returns:
            Dictionary with:
                - 'sound_logits': [B, num_classes]
                - 'sound_probs': [B, num_classes]
                - 'direction': [B, num_directions]
                - 'priority': [B, 1]
                - 'urgency': [B, 1]
        """
        B = audio_features.shape[0]
        
        if audio_features.dim() == 2:
            audio_features = audio_features.unsqueeze(1)  # [B, 1, input_dim]
        
        # Spectrogram branch
        if audio_features.dim() == 3:
            T, freq_bins = audio_features.shape[1], audio_features.shape[2]
            spec = audio_features.unsqueeze(1)  # [B, 1, T, freq_bins]
            
            cnn_out = self.spectrogram_cnn(spec)  # [B, 64, T', embed_dim//4]
            cnn_out = cnn_out.permute(0, 2, 1, 3).contiguous()  # [B, T', 64, embed_dim//4]
            cnn_out = cnn_out.contiguous().reshape(B, cnn_out.shape[1], -1)  # [B, T', 64*(embed_dim//4)]
            
            audio_embed = self.spectrogram_proj(cnn_out)  # [B, T', embed_dim]
            attended, _ = self.temporal_attention(audio_embed, audio_embed, audio_embed)
            audio_embed = self.attention_norm(audio_embed + attended)
            audio_embed = audio_embed.mean(dim=1)  # [B, embed_dim]
        else:
            audio_embed = self.input_proj(audio_features.mean(dim=1))
        
        # Sound classification
        sound_logits = self.classifier(audio_embed)
        sound_probs = F.softmax(sound_logits, dim=1)
        
        # Learnable direction prediction
        direction = self.direction_head(audio_embed)
        
        # Priority and urgency
        priority = self.priority_head(audio_embed)
        
        # Expected urgency: weighted sum over all classes (handles uncertainty)
        urgency_map = getattr(self, 'urgency_map')
        urgency = (sound_probs * urgency_map.unsqueeze(0)).sum(dim=1, keepdim=True)
        
        return {
            'sound_logits': sound_logits,
            'sound_probs': sound_probs,
            'direction': direction,
            'priority': priority,
            'urgency': urgency
        }
