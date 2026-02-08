"""Global Encoder for Multi-Vector Retrieval CLIP ViT-B/32 or DINOv2 for global scene embeddings."""

import torch
import torch.nn as nn
from typing import Optional
import torchvision.transforms as T

# Optional transformers import (for CLIP)
try:
    from transformers import CLIPModel, CLIPProcessor
    import os
    # Use HF_TOKEN if available (reduces rate limiting warnings)
    HF_TOKEN = os.environ.get("HF_TOKEN")
except ImportError:
    CLIPModel = None
    CLIPProcessor = None
    HF_TOKEN = None


class GlobalEncoder(nn.Module):
    """Global encoder using CLIP or DINOv2. Provides global scene-level embeddings for fast retrieval."""
    
    def __init__(
        self,
        model_name: str = 'openai/clip-vit-base-patch32',
        embed_dim: int = 512,
        use_clip: bool = True
    ):
        super().__init__()
        
        self.model_name = model_name
        self.embed_dim = embed_dim
        self.use_clip = use_clip
        
        if use_clip:
            if CLIPModel is None or CLIPProcessor is None:
                raise ImportError("transformers library not found. Install with: pip install transformers")
            # Load CLIP model (use HF_TOKEN if available to avoid rate limiting warnings)
            kwargs = {}
            if HF_TOKEN:
                kwargs['token'] = HF_TOKEN
            self.clip_model = CLIPModel.from_pretrained(model_name, **kwargs)
            self.clip_processor = CLIPProcessor.from_pretrained(model_name, **kwargs)
            
            # CLIP image encoder.
            self.encoder = self.clip_model.vision_model
            
            # Projection to target dimension if needed.
            if embed_dim != self.clip_model.config.vision_config.hidden_size:
                self.proj = nn.Linear(
                    self.clip_model.config.vision_config.hidden_size,
                    embed_dim
                )
            else:
                self.proj = nn.Identity()
        else:
            # DINOv2 (would need to install dinov2 package) For now, placeholder.
            self.encoder = None
            self.proj = nn.Linear(768, embed_dim)  # DINOv2 base dimension.
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to global embeddings. Args: images: Input images [B, 3, H, W] (normalized) Returns: Global embeddings [B, embed_dim]."""
        if self.use_clip and self.encoder is not None:
            # CLIP forward pass.
            outputs = self.encoder(pixel_values=images)
            # Get CLS token or pooled output.
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                global_emb = outputs.pooler_output
            else:
                global_emb = outputs.last_hidden_state[:, 0]  # CLS token.
            
            # Project to target dimension.
            global_emb = self.proj(global_emb)
        else:
            # Placeholder: simple projection.
            B = images.shape[0]
            global_emb = self.proj(torch.randn(B, 768, device=images.device))
        
        # L2 normalize for cosine similarity.
        global_emb = nn.functional.normalize(global_emb, p=2, dim=1)
        
        return global_emb







