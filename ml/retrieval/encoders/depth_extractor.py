"""Depth Extractor for Multi-Vector Retrieval

Uses MiDaS for monocular depth estimation and encodes depth maps."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
try:
    from midas.model_loader import load_model
    MIDAS_AVAILABLE = True
except ImportError:
    MIDAS_AVAILABLE = False


class DepthExtractor(nn.Module):
    """Depth extractor using MiDaS.
    
    Architecture:
    - MiDaS: Monocular depth estimation
    - Depth encoder: CNN to encode depth maps
    - Output: Depth embeddings"""
    
    def __init__(
        self,
        embed_dim: int = 256,
        use_midas: bool = True
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.use_midas = use_midas and MIDAS_AVAILABLE
        
        # MiDaS model (loaded on demand)
        self.midas_model = None
        
        # Depth encoder: CNN to encode depth maps
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, embed_dim)
        )
    
    def _load_midas(self):
        """Load MiDaS model on demand."""
        if self.midas_model is None and self.use_midas:
            try:
                self.midas_model = load_model("DPT_Large")
            except Exception:
                self.use_midas = False
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract depth embeddings.
        
        Args:
            images: Input images [B, 3, H, W]
        
        Returns:
            Depth embeddings [B, embed_dim]"""
        B = images.shape[0]
        
        # Estimate depth
        if self.use_midas:
            self._load_midas()
            if self.midas_model is not None:
                with torch.no_grad():
                    depth_maps = self.midas_model(images)  # [B, 1, H, W]
            else:
                # Fallback: synthetic depth
                depth_maps = self._synthetic_depth(images)
        else:
            # Synthetic depth estimation
            depth_maps = self._synthetic_depth(images)
        
        # Encode depth maps
        depth_embeddings = self.depth_encoder(depth_maps)  # [B, embed_dim]
        
        # L2 normalize
        depth_embeddings = F.normalize(depth_embeddings, p=2, dim=1)
        
        return depth_embeddings
    
    def _synthetic_depth(self, images: torch.Tensor) -> torch.Tensor:
        """Generate synthetic depth map as fallback."""
        # Simple depth estimation based on image intensity
        # Lower intensity = farther (simplified)
        gray = images.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        depth = 1.0 - gray  # Invert: darker = farther
        return depth


