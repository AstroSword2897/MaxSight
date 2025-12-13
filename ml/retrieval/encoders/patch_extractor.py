"""
Patch Extractor for Multi-Vector Retrieval

Extracts ViT patch tokens and clusters them into region groups.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from sklearn.cluster import KMeans
import numpy as np


class PatchExtractor(nn.Module):
    """
    Extracts ViT patch tokens and clusters them.
    
    Architecture:
    - Extract patch tokens from ViT before pooling
    - Cluster tokens into region groups (KMeans or attention pooling)
    - Store patch token embeddings
    """
    
    def __init__(
        self,
        vit_backbone: Optional[nn.Module] = None,
        num_clusters: int = 25,
        embed_dim: int = 768,
        use_kmeans: bool = True
    ):
        super().__init__()
        
        self.vit_backbone = vit_backbone
        self.num_clusters = num_clusters
        self.embed_dim = embed_dim
        self.use_kmeans = use_kmeans
        
        # Attention pooling (alternative to KMeans)
        if not use_kmeans:
            self.attention_pool = nn.MultiheadAttention(
                embed_dim, num_heads=8, batch_first=True
            )
            self.cluster_tokens = nn.Parameter(
                torch.randn(1, num_clusters, embed_dim) * 0.02
            )
    
    def forward(
        self,
        images: torch.Tensor,
        vit_patch_tokens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract and cluster patch tokens.
        
        Args:
            images: Input images [B, 3, H, W]
            vit_patch_tokens: Optional pre-computed patch tokens [B, N_patches, embed_dim]
        
        Returns:
            Clustered patch embeddings [B, num_clusters, embed_dim]
        """
        B = images.shape[0]
        
        # Get patch tokens from ViT if not provided
        if vit_patch_tokens is None and self.vit_backbone is not None:
            _, vit_patch_tokens = self.vit_backbone(images, return_patch_tokens=True)
        
        if vit_patch_tokens is None:
            # Fallback: random tokens
            vit_patch_tokens = torch.randn(B, 196, self.embed_dim, device=images.device)
        
        # Cluster patch tokens
        if self.use_kmeans:
            # KMeans clustering (on CPU for sklearn)
            clustered_embeddings = []
            for b in range(B):
                patches = vit_patch_tokens[b].detach().cpu().numpy()  # [N_patches, embed_dim]
                
                # KMeans clustering
                kmeans = KMeans(n_clusters=self.num_clusters, random_state=0, n_init=10)
                cluster_labels = kmeans.fit_predict(patches)
                cluster_centers = kmeans.cluster_centers_  # [num_clusters, embed_dim]
                
                clustered_embeddings.append(torch.tensor(cluster_centers, device=images.device))
            
            clustered_embeddings = torch.stack(clustered_embeddings)  # [B, num_clusters, embed_dim]
        else:
            # Attention pooling
            cluster_tokens = self.cluster_tokens.expand(B, -1, -1)  # [B, num_clusters, embed_dim]
            clustered_embeddings, _ = self.attention_pool(
                query=cluster_tokens,
                key=vit_patch_tokens,
                value=vit_patch_tokens
            )  # [B, num_clusters, embed_dim]
        
        # L2 normalize
        clustered_embeddings = F.normalize(clustered_embeddings, p=2, dim=2)
        
        return clustered_embeddings


