"""
Scene Description Head for MaxSight 3.0

Transformer decoder for generating natural language scene descriptions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List


class SceneDescriptionHead(nn.Module):
    """
    Scene description head with transformer decoder.
    
    Architecture:
    - Input: Global + region + OCR embeddings
    - Transformer decoder: Generates natural language descriptions
    - Condition-aware: Adjusts verbosity based on vision impairment type
    """
    
    def __init__(
        self,
        global_dim: int = 512,
        region_dim: int = 256,
        ocr_dim: int = 256,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        vocab_size: int = 30000,  # Word vocabulary
        max_length: int = 100,
        condition_modes: Optional[List[str]] = None
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.condition_modes = condition_modes or ['normal', 'cvi', 'glaucoma', 'amd']
        
        # Input projections
        self.global_proj = nn.Linear(global_dim, embed_dim)
        self.region_proj = nn.Linear(region_dim, embed_dim)
        self.ocr_proj = nn.Linear(ocr_dim, embed_dim)
        
        # Condition-aware verbosity controller
        self.verbosity_controller = nn.ModuleDict({
            mode: nn.Linear(embed_dim, embed_dim)
            for mode in self.condition_modes
        })
        
        # Combine inputs
        self.input_fusion = nn.Linear(embed_dim * 3, embed_dim)
        
        # Word embedding
        self.word_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_length, embed_dim) * 0.02
        )
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, vocab_size)
        
        # Special tokens
        self.sos_token_id = 1
        self.eos_token_id = 2
        self.pad_token_id = 0
    
    def forward(
        self,
        global_embedding: torch.Tensor,  # [B, global_dim]
        region_embeddings: torch.Tensor,  # [B, N_regions, region_dim]
        ocr_embeddings: Optional[torch.Tensor] = None,  # [B, N_text, ocr_dim]
        condition_mode: str = 'normal',
        target_text: Optional[torch.Tensor] = None  # [B, seq_len] for training
    ) -> Dict[str, torch.Tensor]:
        """
        Generate scene description.
        
        Args:
            global_embedding: Global scene embedding [B, global_dim]
            region_embeddings: Region embeddings [B, N_regions, region_dim]
            ocr_embeddings: Optional OCR embeddings [B, N_text, ocr_dim]
            condition_mode: Vision condition mode
            target_text: Optional target text for training
        
        Returns:
            Dictionary with:
                - 'description_logits': [B, seq_len, vocab_size]
                - 'description': Generated text (if inference)
        """
        B = global_embedding.shape[0]
        
        # Project inputs
        global_proj = self.global_proj(global_embedding).unsqueeze(1)  # [B, 1, embed_dim]
        region_proj = self.region_proj(region_embeddings)  # [B, N_regions, embed_dim]
        
        if ocr_embeddings is not None:
            ocr_proj = self.ocr_proj(ocr_embeddings)  # [B, N_text, embed_dim]
            ocr_global = ocr_proj.mean(dim=1, keepdim=True)  # [B, 1, embed_dim]
        else:
            ocr_global = torch.zeros(B, 1, self.embed_dim, device=global_embedding.device)
        
        # Combine inputs
        combined = torch.cat([global_proj, region_proj.mean(dim=1, keepdim=True), ocr_global], dim=2)
        memory = self.input_fusion(combined)  # [B, 1, embed_dim]
        
        # Condition-aware verbosity adjustment
        if condition_mode in self.verbosity_controller:
            memory = self.verbosity_controller[condition_mode](memory)
        
        # Decode description
        if target_text is not None:
            # Training: teacher forcing
            tgt = self.word_embedding(target_text)  # [B, seq_len, embed_dim]
            tgt = tgt + self.pos_encoding[:, :target_text.shape[1], :]
            decoded = self.decoder(tgt, memory)  # [B, seq_len, embed_dim]
        else:
            # Inference: autoregressive generation
            decoded_tokens = []
            current_token = torch.full((B, 1), self.sos_token_id, dtype=torch.long, device=global_embedding.device)
            
            for _ in range(self.max_length):
                tgt = self.word_embedding(current_token)  # [B, current_len, embed_dim]
                tgt = tgt + self.pos_encoding[:, :tgt.shape[1], :]
                out = self.decoder(tgt, memory)  # [B, current_len, embed_dim]
                
                # Get next token
                next_logits = self.output_proj(out[:, -1, :])  # [B, vocab_size]
                next_token = next_logits.argmax(dim=1, keepdim=True)  # [B, 1]
                
                # Stop if EOS token
                if (next_token == self.eos_token_id).all():
                    break
                
                decoded_tokens.append(out[:, -1:, :])
                current_token = torch.cat([current_token, next_token], dim=1)
            
            if decoded_tokens:
                decoded = torch.cat(decoded_tokens, dim=1)  # [B, generated_len, embed_dim]
            else:
                decoded = out  # Fallback
        
        # Output logits
        description_logits = self.output_proj(decoded)  # [B, seq_len, vocab_size]
        
        return {
            'description_logits': description_logits,
            'description': decoded
        }


