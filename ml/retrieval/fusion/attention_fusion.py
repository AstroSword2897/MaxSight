"""
Attention-Based Fusion for Multi-Vector Retrieval

Query-adaptive attention fusion of multiple embedding types.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List


class AttentionFusion(nn.Module):
    """
    Query-adaptive attention fusion.
    
    Learns attention weights per embedding type dynamically based on query.
    """
    
    def __init__(
        self,
        embedding_dims: Dict[str, int],  # {'global': 512, 'region': 256, ...}
        fused_dim: int = 512,
        num_heads: int = 8
    ):
        super().__init__()
        
        self.embedding_dims = embedding_dims
        self.fused_dim = fused_dim
        
        # Projections for each embedding type
        self.projections = nn.ModuleDict({
            name: nn.Linear(dim, fused_dim)
            for name, dim in embedding_dims.items()
        })
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            fused_dim, num_heads, batch_first=True
        )
        
        # Query generator (learns what to attend to)
        self.query_generator = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.ReLU(),
            nn.Linear(fused_dim, fused_dim)
        )
        
        # Output projection
        self.output_proj = nn.Linear(fused_dim, fused_dim)
        self.norm = nn.LayerNorm(fused_dim)
    
    def forward(
        self,
        embeddings: Dict[str, torch.Tensor],
        query_embedding: Optional[torch.Tensor] = None  # [B, fused_dim]
    ) -> torch.Tensor:
        """
        Fuse multiple embeddings using attention.
        
        Args:
            embeddings: Dictionary of embeddings {name: tensor}
            query_embedding: Optional query embedding for adaptive fusion
        
        Returns:
            Fused embedding [B, fused_dim]
        """
        B = next(iter(embeddings.values())).shape[0]
        device = next(iter(embeddings.values())).device
        
        # Project all embeddings to common dimension
        projected_embeddings = []
        for name, emb in embeddings.items():
            if name in self.projections:
                proj_emb = self.projections[name](emb)  # [B, ..., fused_dim]
                
                # Handle different shapes
                if proj_emb.dim() == 2:
                    # [B, fused_dim] -> [B, 1, fused_dim]
                    proj_emb = proj_emb.unsqueeze(1)
                elif proj_emb.dim() == 3:
                    # [B, N, fused_dim] - keep as is
                    pass
                else:
                    # Flatten extra dimensions
                    proj_emb = proj_emb.reshape(B, -1, self.fused_dim)
                
                projected_embeddings.append(proj_emb)
        
        # Concatenate all embeddings
        all_embeddings = torch.cat(projected_embeddings, dim=1)  # [B, N_total, fused_dim]
        
        # Generate query
        if query_embedding is not None:
            query = self.query_generator(query_embedding).unsqueeze(1)  # [B, 1, fused_dim]
        else:
            # Use mean of embeddings as query
            query = all_embeddings.mean(dim=1, keepdim=True)  # [B, 1, fused_dim]
            query = self.query_generator(query.squeeze(1)).unsqueeze(1)
        
        # Attention: query attends to all embeddings
        fused, attn_weights = self.attention(
            query=query,
            key=all_embeddings,
            value=all_embeddings
        )  # [B, 1, fused_dim]
        
        fused = fused.squeeze(1)  # [B, fused_dim]
        fused = self.norm(fused)
        fused = self.output_proj(fused)
        
        # L2 normalize
        fused = F.normalize(fused, p=2, dim=1)
        
        return fused


