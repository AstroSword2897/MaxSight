"""
Transformer-Based OCR Head for MaxSight 3.0

Transformer encoder for text detection and decoder for text recognition.
Supports per-region decoding, optional context embeddings, and scene-text integration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class TransformerOCRHead(nn.Module):
    """
    Transformer-based OCR head.

    Features:
    - Transformer encoder: Text region features -> embeddings
    - Transformer decoder: autoregressive text recognition per region
    - Optional context embeddings for scene objects
    - Text bounding box prediction
    """

    def __init__(
        self,
        input_dim: int = 256,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        vocab_size: int = 10000,
        max_text_length: int = 50
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.max_text_length = max_text_length

        # Input projection
        self.input_proj = nn.Linear(input_dim, embed_dim)

        # Positional encoding for regions
        self.pos_encoding = nn.Parameter(torch.randn(1, 500, embed_dim) * 0.02)  # max 500 regions

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        # Character embedding
        self.char_embedding = nn.Embedding(vocab_size, embed_dim)
        self.sos_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # learned SOS

        # Output projection
        self.output_proj = nn.Linear(embed_dim, vocab_size)

        # Text region detection head
        self.text_detection_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 4)  # bounding box coordinates
        )

    def forward(
        self,
        features: torch.Tensor,               # [B, N_regions, input_dim]
        context_embeddings: Optional[torch.Tensor] = None,  # [B, N_objects, embed_dim]
        target_text: Optional[torch.Tensor] = None          # [B, N_regions, seq_len]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through OCR head.

        Args:
            features: Text region features [B, N_regions, input_dim]
            context_embeddings: Optional object context [B, N_objects, embed_dim]
            target_text: Optional target text for teacher forcing [B, N_regions, seq_len]

        Returns:
            Dictionary with:
                - 'text_logits': [B, N_regions, max_length, vocab_size]
                - 'text_boxes': [B, N_regions, 4]
                - 'text_scores': [B, N_regions]
                - 'encoded_features': [B, N_regions, embed_dim]
        """
        B, N_regions, _ = features.shape

        # Project input features
        x = self.input_proj(features)  # [B, N_regions, embed_dim]

        # Add positional encoding
        x = x + self.pos_encoding[:, :N_regions, :]

        # Integrate optional context embeddings
        if context_embeddings is not None:
            context_mean = context_embeddings.mean(dim=1, keepdim=True)  # [B,1,embed_dim]
            x = x + context_mean

        # Transformer encoder
        encoded = self.encoder(x)  # [B, N_regions, embed_dim]

        # Text detection
        text_boxes = self.text_detection_head(encoded)  # [B, N_regions, 4]

        # Text recognition per region
        all_logits = []
        for r in range(N_regions):
            region_encoded = encoded[:, r:r+1, :]  # [B,1,embed_dim]

            if target_text is not None:
                # Training: teacher forcing
                tgt_seq = target_text[:, r, :]  # [B, seq_len]
                tgt_emb = self.char_embedding(tgt_seq)  # [B, seq_len, embed_dim]
                decoded = self.decoder(tgt_emb, region_encoded)
            else:
                # Inference: autoregressive
                decoded_tokens = []
                sos = self.sos_token.expand(B, -1, -1)  # [B,1,embed_dim]
                for _ in range(self.max_text_length):
                    tgt_input = torch.cat(decoded_tokens, dim=1) if decoded_tokens else sos
                    out = self.decoder(tgt_input, region_encoded)
                    decoded_tokens.append(out[:, -1:, :])  # append last step
                decoded = torch.cat(decoded_tokens, dim=1)  # [B, max_length, embed_dim]

            logits = self.output_proj(decoded)  # [B, seq_len/max_length, vocab_size]
            all_logits.append(logits.unsqueeze(1))  # add region dim

        text_logits = torch.cat(all_logits, dim=1)  # [B, N_regions, max_length, vocab_size]

        # Text scores: mean confidence per region
        text_probs = F.softmax(text_logits, dim=-1)
        text_scores = text_probs.max(dim=-1)[0].mean(dim=-1)  # [B, N_regions]

        return {
            'text_logits': text_logits,
            'text_boxes': text_boxes,
            'text_scores': text_scores,
            'encoded_features': encoded
        }
