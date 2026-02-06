"""Patch Extractor for Multi-Vector Retrieval..."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
# Optional sklearn import (only for offline preprocessing)
try:
    from sklearn.cluster import KMeans
except ImportError:
    KMeans = None


class PatchExtractor(nn.Module):
    """Fully differentiable patch extractor with attention-based clustering...."""
    
    def __init__(
        self,
        vit_backbone: Optional[nn.Module] = None,
        num_clusters: int = 25,
        embed_dim: int = 768,
        num_heads: int = 8,
        use_soft_kmeans: bool = False,  # NEW: Differentiable soft KMeans alternative
        temperature: float = 0.1,  # For soft KMeans
        use_gradient_checkpointing: bool = False  # NEW: Memory efficiency
    ):
        super().__init__()
        
        self.vit_backbone = vit_backbone
        self.num_clusters = num_clusters
        self.embed_dim = embed_dim
        self.use_soft_kmeans = use_soft_kmeans
        self.temperature = temperature
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # FIXED: Always use attention pooling (fully differentiable, GPU-efficient)
        self.attention_pool = nn.MultiheadAttention(
            embed_dim, num_heads=num_heads, batch_first=True
        )
        
        # Learnable cluster tokens (query vectors)
        self.cluster_tokens = nn.Parameter(
            torch.randn(1, num_clusters, embed_dim) * 0.02
        )
        
        # NEW: Optional projection for better clustering
        self.patch_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # NEW: Soft KMeans alternative (fully differentiable)
        if use_soft_kmeans:
            # Learnable cluster centers
            self.soft_cluster_centers = nn.Parameter(
                torch.randn(1, num_clusters, embed_dim) * 0.02
            )
    
    def _attention_pooling(
        self,
        patch_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Attention-based pooling (fully differentiable, GPU-efficient)...."""
        B = patch_tokens.shape[0]
        
        # Project patches for better clustering
        patch_tokens_proj = self.patch_proj(patch_tokens)  # [B, N_patches, embed_dim]
        
        # Expand cluster tokens for batch
        cluster_tokens = self.cluster_tokens.expand(B, -1, -1)  # [B, num_clusters, embed_dim]
        
        # Attention pooling: cluster tokens attend to patch tokens
        clustered_embeddings, attention_weights = self.attention_pool(
            query=cluster_tokens,  # [B, num_clusters, embed_dim]
            key=patch_tokens_proj,  # [B, N_patches, embed_dim]
            value=patch_tokens_proj  # [B, N_patches, embed_dim]
        )  # [B, num_clusters, embed_dim]
        
        return clustered_embeddings
    
    def _soft_kmeans(
        self,
        patch_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Soft KMeans clustering (fully differentiable alternative to hard KMeans)...."""
        B, N_patches, _ = patch_tokens.shape
        
        # Project patches
        patch_tokens_proj = self.patch_proj(patch_tokens)  # [B, N_patches, embed_dim]
        
        # Expand cluster centers
        cluster_centers = self.soft_cluster_centers.expand(B, -1, -1)  # [B, num_clusters, embed_dim]
        
        # Compute distances: [B, N_patches, num_clusters]
        # Use cosine similarity (normalized) for better clustering
        patch_norm = F.normalize(patch_tokens_proj, p=2, dim=2)  # [B, N_patches, embed_dim]
        center_norm = F.normalize(cluster_centers, p=2, dim=2)  # [B, num_clusters, embed_dim]
        
        # Cosine similarity: [B, N_patches, num_clusters]
        similarities = torch.bmm(
            patch_norm,  # [B, N_patches, embed_dim]
            center_norm.transpose(1, 2)  # [B, embed_dim, num_clusters]
        )  # [B, N_patches, num_clusters]
        
        # Soft assignment with temperature: [B, N_patches, num_clusters]
        soft_assignments = F.softmax(similarities / self.temperature, dim=2)
        
        # Weighted average: [B, num_clusters, embed_dim]
        clustered_embeddings = torch.bmm(
            soft_assignments.transpose(1, 2),  # [B, num_clusters, N_patches]
            patch_tokens_proj  # [B, N_patches, embed_dim]
        )  # [B, num_clusters, embed_dim]
        
        return clustered_embeddings
    
    def forward(
        self,
        images: torch.Tensor,
        vit_patch_tokens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Extract and cluster patch tokens (fully differentiable, GPU-efficient)...."""
        B = images.shape[0]
        device = images.device
        
        # FIXED: Get patch tokens from ViT if not provided
        if vit_patch_tokens is None:
            if self.vit_backbone is not None:
                # Extract from ViT backbone
                try:
                    _, vit_patch_tokens = self.vit_backbone(images, return_patch_tokens=True)
                except (AttributeError, TypeError):
                    # Fallback: try standard forward and extract intermediate
                    vit_outputs = self.vit_backbone(images)
                    if isinstance(vit_outputs, tuple):
                        vit_patch_tokens = vit_outputs[1] if len(vit_outputs) > 1 else None
                    else:
                        vit_patch_tokens = None
            else:
                vit_patch_tokens = None
        
        # FIXED: Proper error handling (no hard-coded fallback)
        if vit_patch_tokens is None:
            raise ValueError(
                "vit_patch_tokens must be provided or vit_backbone must be set. "
                "Cannot proceed without patch tokens."
            )
        
        # Validate shape
        if vit_patch_tokens.dim() != 3:
            raise ValueError(
                f"Expected 3D tensor [B, N_patches, embed_dim], got {vit_patch_tokens.shape}"
            )
        
        if vit_patch_tokens.shape[2] != self.embed_dim:
            raise ValueError(
                f"Expected embed_dim={self.embed_dim}, got {vit_patch_tokens.shape[2]}"
            )
        
        # FIXED: Fully differentiable clustering (all on GPU, no batch loops)
        if self.use_soft_kmeans:
            # Soft KMeans (differentiable alternative)
            clustered_embeddings = self._soft_kmeans(vit_patch_tokens)
        else:
            # Attention pooling (default, fully differentiable)
            if self.use_gradient_checkpointing and self.training:
                # Memory-efficient: checkpoint attention during training
                clustered_embeddings = torch.utils.checkpoint.checkpoint(
                    self._attention_pooling,
                    vit_patch_tokens,
                    use_reentrant=False
                )
            else:
                clustered_embeddings = self._attention_pooling(vit_patch_tokens)
        
        # FIXED: Safe L2 normalization (handle zero vectors)
        # Add small epsilon to prevent NaNs
        clustered_embeddings = F.normalize(
            clustered_embeddings + 1e-8,  # Prevent zero vectors
            p=2,
            dim=2
        )
        
        # Final validation
        if torch.isnan(clustered_embeddings).any() or torch.isinf(clustered_embeddings).any():
            raise RuntimeError("NaN/Inf detected in clustered embeddings")
        
        return clustered_embeddings
    
    def get_attention_weights(
        self,
        images: torch.Tensor,
        vit_patch_tokens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get attention weights for visualization/debugging.
        
        Returns:
            Attention weights [B, num_clusters, N_patches]"""
        if vit_patch_tokens is None and self.vit_backbone is not None:
            _, vit_patch_tokens = self.vit_backbone(images, return_patch_tokens=True)
        
        if vit_patch_tokens is None:
            raise ValueError("vit_patch_tokens must be provided")
        
        B = vit_patch_tokens.shape[0]
        patch_tokens_proj = self.patch_proj(vit_patch_tokens)
        cluster_tokens = self.cluster_tokens.expand(B, -1, -1)
        
        _, attention_weights = self.attention_pool(
            query=cluster_tokens,
            key=patch_tokens_proj,
            value=patch_tokens_proj
        )
        
        return attention_weights  # [B, num_clusters, N_patches]


