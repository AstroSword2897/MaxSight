"""
Sound Event Classification Head for MaxSight 3.0

CNN + temporal attention for sound event classification and directional detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List


class SoundEventHead(nn.Module):
    """
    Sound event classification head.
    
    Architecture:
    - CNN on spectrograms
    - Temporal attention for sound events
    - Directional audio detection
    - Priority weighting for urgent events
    """
    
    def __init__(
        self,
        input_dim: int = 128,  # Spectrogram or MFCC features
        num_classes: int = 15,  # Sound classes
        embed_dim: int = 256,
        num_heads: int = 8,
        num_directions: int = 4  # left, right, front, back
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_directions = num_directions
        
        # Spectrogram CNN (if input is spectrogram)
        self.spectrogram_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((None, embed_dim // 4))
        )
        
        # Temporal attention
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        
        # Sound classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # Directional detection head
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
            nn.Sigmoid()  # Priority score [0, 1]
        )
        
        # Urgency mapping (sound class -> urgency level)
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
        audio_features: torch.Tensor,  # [B, T, input_dim] or [B, input_dim]
        stereo_channels: Optional[torch.Tensor] = None  # [B, T, 2]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through sound event head.
        
        Args:
            audio_features: Audio features [B, T, input_dim] or [B, input_dim]
            stereo_channels: Optional stereo channels [B, T, 2]
        
        Returns:
            Dictionary with:
                - 'sound_logits': [B, num_classes]
                - 'sound_probs': [B, num_classes]
                - 'direction': [B, num_directions]
                - 'priority': [B, 1]
                - 'urgency': [B, 1]
        """
        B = audio_features.shape[0]
        
        # Handle different input formats
        if audio_features.dim() == 2:
            # [B, input_dim] - single frame
            audio_features = audio_features.unsqueeze(1)  # [B, 1, input_dim]
        
        # If spectrogram format [B, T, F], convert for CNN
        if audio_features.dim() == 3:
            T, F = audio_features.shape[1], audio_features.shape[2]
            # Reshape for CNN: [B, T, F] -> [B, 1, T, F]
            spec = audio_features.unsqueeze(1)
            
            # Apply CNN
            cnn_out = self.spectrogram_cnn(spec)  # [B, 64, T', embed_dim//4]
            # Flatten: [B, 64, T', embed_dim//4] -> [B, T', embed_dim]
            cnn_out = cnn_out.permute(0, 2, 1, 3).contiguous()
            cnn_out = cnn_out.reshape(B, -1, self.spectrogram_cnn[0].out_channels * (self.spectrogram_cnn[-1].output_size[1] if hasattr(self.spectrogram_cnn[-1], 'output_size') else 64))
            
            # Project to embed_dim if needed
            if cnn_out.shape[2] != embed_dim:
                proj = nn.Linear(cnn_out.shape[2], embed_dim).to(cnn_out.device)
                cnn_out = proj(cnn_out)
            
            # Temporal attention
            attended, _ = self.temporal_attention(cnn_out, cnn_out, cnn_out)
            audio_embed = (cnn_out + attended).mean(dim=1)  # [B, embed_dim]
        else:
            # Simple projection
            audio_embed = nn.Linear(audio_features.shape[-1], embed_dim).to(audio_features.device)(audio_features.mean(dim=1))
        
        # Sound classification
        sound_logits = self.classifier(audio_embed)  # [B, num_classes]
        sound_probs = F.softmax(sound_logits, dim=1)  # [B, num_classes]
        
        # Directional detection
        if stereo_channels is not None:
            # Compute direction from stereo
            left = stereo_channels[:, :, 0].mean(dim=1)  # [B]
            right = stereo_channels[:, :, 1].mean(dim=1)  # [B]
            ild = left - right
            
            direction = torch.stack([
                (ild < -0.1).float(),  # Left
                (ild > 0.1).float(),   # Right
                (torch.abs(ild) < 0.1).float(),  # Front
                torch.zeros(B, device=audio_features.device)  # Back
            ], dim=1)
            direction = direction / (direction.sum(dim=1, keepdim=True) + 1e-8)
        else:
            direction = self.direction_head(audio_embed)  # [B, num_directions]
        
        # Priority weighting
        priority = self.priority_head(audio_embed)  # [B, 1]
        
        # Urgency level (based on predicted sound class)
        predicted_class = sound_logits.argmax(dim=1)  # [B]
        urgency = self.urgency_map[predicted_class].unsqueeze(1).float()  # [B, 1]
        
        return {
            'sound_logits': sound_logits,
            'sound_probs': sound_probs,
            'direction': direction,
            'priority': priority,
            'urgency': urgency
        }


