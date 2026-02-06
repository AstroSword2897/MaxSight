"""Meta-Learning Fusion Weights for Phase 6: Personalization & Active Guidance..."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class UserProfile:
    """User preference profile for personalization."""
    user_id: str
    preferred_modalities: List[str]  # ['vision', 'audio', 'haptic']
    task_preferences: Dict[str, float]  # Task -> preference weight
    adaptation_rate: float = 0.1  # How quickly to adapt


class MetaFusionWeights(nn.Module):
    """Meta-learning fusion weights that adapt to user preferences.
    
    Uses gradient-based meta-learning to quickly adapt fusion weights
    based on user feedback and task performance."""
    
    def __init__(
        self,
        num_modalities: int = 3,  # vision, audio, haptic
        embed_dim: int = 256,
        hidden_dim: int = 128,
        adaptation_steps: int = 3
    ):
        super().__init__()
        self.num_modalities = num_modalities
        self.embed_dim = embed_dim
        self.adaptation_steps = adaptation_steps
        
        # Base fusion weights (learned)
        self.base_weights = nn.Parameter(torch.ones(num_modalities) / num_modalities)
        
        # Meta-learner: predicts adaptation from user profile
        self.meta_learner = nn.Sequential(
            nn.Linear(num_modalities + 4, hidden_dim),  # modalities + task_type + urgency + confidence + user_embed
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_modalities),
            nn.Softmax(dim=-1)
        )
        
        # User profile embeddings
        self.user_embedding = nn.Embedding(1000, hidden_dim)  # Support up to 1000 users
        
    def forward(
        self,
        modality_embeddings: Dict[str, torch.Tensor],
        user_id: Optional[torch.Tensor] = None,
        task_type: Optional[str] = None,
        urgency: Optional[float] = None,
        confidence: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute personalized fusion weights...."""
        B = next(iter(modality_embeddings.values())).shape[0]
        device = next(iter(modality_embeddings.values())).device
        
        # Get base weights
        base_weights = self.base_weights.unsqueeze(0).expand(B, -1)  # [B, num_modalities]
        
        # Build meta-learner input
        meta_input = []
        
        # Modality presence (1 if available, 0 if not)
        modality_presence = torch.zeros(B, self.num_modalities, device=device)
        modality_names = ['vision', 'audio', 'haptic']
        for i, name in enumerate(modality_names):
            if name in modality_embeddings:
                modality_presence[:, i] = 1.0
        
        meta_input.append(modality_presence)
        
        # Task type encoding
        task_encoding = torch.zeros(B, 1, device=device)
        if task_type == 'navigation':
            task_encoding.fill_(1.0)
        elif task_type == 'reading':
            task_encoding.fill_(0.5)
        elif task_type == 'therapy':
            task_encoding.fill_(0.0)
        meta_input.append(task_encoding)
        
        # Urgency (default 0.5 if not provided)
        urgency_tensor = torch.full((B, 1), urgency if urgency is not None else 0.5, device=device)
        meta_input.append(urgency_tensor)
        
        # Confidence (default 0.5 if not provided)
        confidence_tensor = torch.full((B, 1), confidence if confidence is not None else 0.5, device=device)
        meta_input.append(confidence_tensor)
        
        # User embedding (if provided)
        if user_id is not None:
            user_emb = self.user_embedding(user_id)  # [B, hidden_dim]
            # Project to single value for meta input
            user_proj = user_emb.mean(dim=1, keepdim=True)  # [B, 1]
            meta_input.append(user_proj)
        else:
            meta_input.append(torch.zeros(B, 1, device=device))
        
        # Concatenate meta inputs
        meta_features = torch.cat(meta_input, dim=1)  # [B, num_modalities + 3 + 1]
        
        # Predict adaptation
        adaptation = self.meta_learner(meta_features)  # [B, num_modalities]
        
        # Combine base weights with adaptation
        fusion_weights = base_weights * 0.7 + adaptation * 0.3
        fusion_weights = F.softmax(fusion_weights, dim=-1)  # Normalize
        
        # Apply fusion
        modality_list = []
        weight_list = []
        
        for i, name in enumerate(modality_names):
            if name in modality_embeddings:
                modality_list.append(modality_embeddings[name])
                weight_list.append(fusion_weights[:, i:i+1])
        
        if not modality_list:
            # Fallback: return vision if available, else zeros
            if 'vision' in modality_embeddings:
                return modality_embeddings['vision'], fusion_weights
            else:
                return torch.zeros(B, self.embed_dim, device=device), fusion_weights
        
        # Weighted fusion
        weighted_embeddings = [emb * w for emb, w in zip(modality_list, weight_list)]
        fused_embedding = sum(weighted_embeddings)
        
        return fused_embedding, fusion_weights
    
    def adapt_to_user(
        self,
        user_profile: UserProfile,
        task_performance: Dict[str, float],
        num_steps: int = 5
    ) -> torch.Tensor:
        """Adapt fusion weights to a specific user using meta-learning...."""
        # Create user-specific adaptation
        # This is a simplified version - full MAML would require inner loop optimization
        
        # Start with base weights
        adapted_weights = self.base_weights.clone()
        
        # Adjust based on preferred modalities
        for i, modality in enumerate(['vision', 'audio', 'haptic']):
            if modality in user_profile.preferred_modalities:
                adapted_weights[i] *= 1.2
        
        # Normalize
        adapted_weights = F.softmax(adapted_weights, dim=0)
        
        return adapted_weights


class ActiveSceneExploration(nn.Module):
    """Active scene exploration for Phase 6.
    
    Determines which regions of the scene to explore next based on:
    - Current uncertainty
    - User preferences
    - Task requirements"""
    
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.uncertainty_threshold = 0.5
        
        # Exploration policy network
        self.exploration_policy = nn.Sequential(
            nn.Linear(embed_dim + 3, 128),  # embedding + uncertainty + urgency + user_pref
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        region_embeddings: torch.Tensor,  # [B, N_regions, embed_dim]
        uncertainties: torch.Tensor,  # [B, N_regions]
        urgency: Optional[torch.Tensor] = None,  # [B]
        user_preference: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Determine which regions to explore next...."""
        B, N, D = region_embeddings.shape
        device = region_embeddings.device
        
        # Build policy input
        policy_inputs = []
        
        # Region embeddings (mean pooled)
        region_features = region_embeddings.mean(dim=1)  # [B, embed_dim]
        policy_inputs.append(region_features)
        
        # Mean uncertainty
        mean_uncertainty = uncertainties.mean(dim=1, keepdim=True)  # [B, 1]
        policy_inputs.append(mean_uncertainty)
        
        # Urgency (default 0.5)
        urgency_tensor = urgency if urgency is not None else torch.full((B, 1), 0.5, device=device)
        policy_inputs.append(urgency_tensor)
        
        # User preference (default 0.5)
        pref_tensor = torch.full((B, 1), user_preference if user_preference is not None else 0.5, device=device)
        policy_inputs.append(pref_tensor)
        
        # Concatenate
        policy_input = torch.cat(policy_inputs, dim=1)  # [B, embed_dim + 3]
        
        # Predict exploration scores
        exploration_scores = self.exploration_policy(policy_input)  # [B, 1]
        
        # Expand to per-region scores
        exploration_scores = exploration_scores.expand(-1, N)  # [B, N_regions]
        
        # Combine with uncertainty (explore uncertain regions)
        combined_scores = exploration_scores * (1.0 + uncertainties)
        
        # Select top-K regions
        K = min(5, N)  # Explore top 5 regions
        _, selected_indices = torch.topk(combined_scores, K, dim=1)  # [B, K]
        
        return combined_scores, selected_indices


class PredictiveNavigationGuidance(nn.Module):
    """Predictive navigation guidance for Phase 6...."""
    
    def __init__(self, embed_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        
        # Path prediction network
        self.path_predictor = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),  # current + goal embeddings
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # [dx, dy, distance, confidence]
        )
        
        # Guidance generator
        self.guidance_generator = nn.Sequential(
            nn.Linear(embed_dim + 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Guidance priority score
        )
        
    def forward(
        self,
        current_embedding: torch.Tensor,  # [B, embed_dim]
        goal_embedding: torch.Tensor,  # [B, embed_dim]
        scene_context: Optional[torch.Tensor] = None  # [B, embed_dim]
    ) -> Dict[str, torch.Tensor]:
        """Predict navigation path and generate guidance...."""
        B = current_embedding.shape[0]
        device = current_embedding.device
        
        # Combine current and goal embeddings
        combined = torch.cat([current_embedding, goal_embedding], dim=1)  # [B, embed_dim * 2]
        
        # Predict path
        path_pred = self.path_predictor(combined)  # [B, 4]
        
        direction = path_pred[:, :2]  # [B, 2]
        distance = path_pred[:, 2:3]  # [B, 1]
        confidence = torch.sigmoid(path_pred[:, 3:4])  # [B, 1]
        
        # Generate guidance priority
        if scene_context is not None:
            guidance_input = torch.cat([scene_context, path_pred], dim=1)  # [B, embed_dim + 4]
        else:
            guidance_input = torch.cat([current_embedding, path_pred], dim=1)
        
        guidance_priority = self.guidance_generator(guidance_input)  # [B, 1]
        guidance_priority = torch.sigmoid(guidance_priority)
        
        return {
            'direction': direction,
            'distance': distance,
            'confidence': confidence,
            'guidance_priority': guidance_priority
        }

