"""
Personalization Head for MaxSight 3.0

Learns user-specific patterns for attention adjustment, verbosity, and alert priorities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class UserPreferences:
    """User preference settings."""
    attention_weights: torch.Tensor  # [num_features]
    verbosity_level: int  # 0-3
    alert_priority_weights: torch.Tensor  # [num_alert_types]
    preferred_output_channel: str  # 'audio', 'visual', 'haptic'


class PersonalizationHead(nn.Module):
    """
    Personalization head for user-specific adaptation.
    
    Learns:
    - Attention adjustment (what user focuses on)
    - Description verbosity preferences
    - Alert priority preferences
    - Online learning from user interactions
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        num_features: int = 10,  # Number of attention features
        num_alert_types: int = 5,
        embed_dim: int = 256
    ):
        super().__init__()
        
        self.num_features = num_features
        self.num_alert_types = num_alert_types
        
        # User embedding (learned per user)
        self.user_embedding = nn.Parameter(torch.randn(1, embed_dim) * 0.02)
        
        # Attention adjustment network
        self.attention_adjuster = nn.Sequential(
            nn.Linear(input_dim + embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_features),
            nn.Softmax(dim=1)
        )
        
        # Verbosity predictor
        self.verbosity_predictor = nn.Sequential(
            nn.Linear(input_dim + embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 4),  # 4 verbosity levels (0-3)
            nn.Softmax(dim=1)
        )
        
        # Alert priority adjuster
        self.alert_priority_adjuster = nn.Sequential(
            nn.Linear(input_dim + embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_alert_types),
            nn.Softmax(dim=1)
        )
        
        # Online learning: adaptation network
        self.adaptation_network = nn.Sequential(
            nn.Linear(embed_dim + 64, embed_dim),  # user_embed + interaction_features
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def forward(
        self,
        scene_features: torch.Tensor,  # [B, input_dim]
        user_id: Optional[int] = None,
        interaction_features: Optional[torch.Tensor] = None  # [B, 64] user interaction history
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through personalization head.
        
        Args:
            scene_features: Scene features [B, input_dim]
            user_id: Optional user ID for user-specific embedding
            interaction_features: Optional interaction history [B, 64]
        
        Returns:
            Dictionary with personalized preferences
        """
        B = scene_features.shape[0]
        
        # Get user embedding (for now, use shared embedding)
        # In practice, would use user_id to index into user embedding table
        user_emb = self.user_embedding.expand(B, -1)  # [B, embed_dim]
        
        # Online adaptation (if interaction features provided)
        if interaction_features is not None:
            combined = torch.cat([user_emb, interaction_features], dim=1)
            user_emb = self.adaptation_network(combined)  # [B, embed_dim]
        
        # Combine scene and user features
        combined_features = torch.cat([scene_features, user_emb], dim=1)  # [B, input_dim + embed_dim]
        
        # Attention adjustment
        attention_weights = self.attention_adjuster(combined_features)  # [B, num_features]
        
        # Verbosity prediction
        verbosity_logits = self.verbosity_predictor(combined_features)  # [B, 4]
        verbosity_level = verbosity_logits.argmax(dim=1)  # [B]
        
        # Alert priority adjustment
        alert_priority_weights = self.alert_priority_adjuster(combined_features)  # [B, num_alert_types]
        
        return {
            'attention_weights': attention_weights,
            'verbosity_level': verbosity_level,
            'verbosity_logits': verbosity_logits,
            'alert_priority_weights': alert_priority_weights,
            'user_embedding': user_emb
        }
    
    def update_user_preferences(
        self,
        user_id: int,
        interaction_features: torch.Tensor,
        feedback: Dict[str, torch.Tensor]
    ):
        """
        Update user preferences based on interaction feedback.
        
        Args:
            user_id: User ID
            interaction_features: Interaction features [64]
            feedback: Feedback dictionary with preferences
        """
        # In practice, this would update user-specific embeddings
        # For now, this is a placeholder for online learning
        pass


