"""
Cross-Task Attention for MaxSight 3.0

Links embeddings between different tasks:
- OCR → Detection (sign → door relationship)
- Detection → Description (objects → natural language)
- Description → OCR (context improves text recognition)
"""

import torch
import torch.nn as nn
from typing import Tuple


class CrossTaskAttention(nn.Module):
    """
    Cross-task attention module.
    
    Enables information sharing between detection, OCR, and description tasks.
    """
    
    def __init__(
        self,
        detection_dim: int,
        ocr_dim: int,
        description_dim: int,
        embed_dim: int = 512,
        num_heads: int = 8
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Projection layers to common dimension
        self.detection_proj = nn.Linear(detection_dim, embed_dim)
        self.ocr_proj = nn.Linear(ocr_dim, embed_dim)
        self.description_proj = nn.Linear(description_dim, embed_dim)
        
        # Cross-attention modules
        # OCR → Detection (sign → door relationship)
        self.ocr_to_detection = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        
        # Detection → Description (objects → natural language)
        self.detection_to_description = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        
        # Description → OCR (context improves text recognition)
        self.description_to_ocr = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        
        # Layer norms
        self.norm_detection = nn.LayerNorm(embed_dim)
        self.norm_ocr = nn.LayerNorm(embed_dim)
        self.norm_description = nn.LayerNorm(embed_dim)
    
    def forward(
        self,
        detection_features: torch.Tensor,  # [B, N_detections, detection_dim]
        ocr_features: torch.Tensor,        # [B, N_text_regions, ocr_dim]
        description_context: torch.Tensor   # [B, description_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through cross-task attention.
        
        Args:
            detection_features: Detection embeddings [B, N_det, detection_dim]
            ocr_features: OCR text embeddings [B, N_text, ocr_dim]
            description_context: Description context [B, description_dim]
        
        Returns:
            det_enhanced: Enhanced detection features [B, N_det, embed_dim]
            ocr_enhanced: Enhanced OCR features [B, N_text, embed_dim]
            desc_enhanced: Enhanced description context [B, embed_dim]
        """
        # Project to common dimension
        det_proj = self.detection_proj(detection_features)  # [B, N_det, embed_dim]
        ocr_proj = self.ocr_proj(ocr_features)              # [B, N_text, embed_dim]
        desc_proj = self.description_proj(description_context).unsqueeze(1)  # [B, 1, embed_dim]
        
        # OCR → Detection: Use OCR context to enhance detection
        det_enhanced, _ = self.ocr_to_detection(
            query=det_proj,
            key=ocr_proj,
            value=ocr_proj
        )  # [B, N_det, embed_dim]
        det_enhanced = self.norm_detection(det_proj + det_enhanced)
        
        # Detection → Description: Use detections to generate descriptions
        desc_enhanced, _ = self.detection_to_description(
            query=desc_proj,
            key=det_enhanced,
            value=det_enhanced
        )  # [B, 1, embed_dim]
        desc_enhanced = self.norm_description(desc_proj + desc_enhanced)
        
        # Description → OCR: Use description context to improve OCR
        ocr_enhanced, _ = self.description_to_ocr(
            query=ocr_proj,
            key=desc_enhanced.expand(-1, ocr_proj.shape[1], -1),
            value=desc_enhanced.expand(-1, ocr_proj.shape[1], -1)
        )  # [B, N_text, embed_dim]
        ocr_enhanced = self.norm_ocr(ocr_proj + ocr_enhanced)
        
        return det_enhanced, ocr_enhanced, desc_enhanced.squeeze(1)


