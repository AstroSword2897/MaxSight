"""
OCR Encoder for Multi-Vector Retrieval

Encodes OCR text snippets using sentence-transformers.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class OCREncoder(nn.Module):
    """
    OCR encoder for text embeddings.
    
    Uses sentence-transformers to embed OCR text snippets.
    """
    
    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        embed_dim: int = 384,
        max_texts: int = 10
    ):
        super().__init__()
        
        self.model_name = model_name
        self.embed_dim = embed_dim
        self.max_texts = max_texts
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.text_encoder = SentenceTransformer(model_name)
                self.use_sentence_transformers = True
            except Exception:
                self.use_sentence_transformers = False
                self.text_encoder = None
        else:
            self.use_sentence_transformers = False
            self.text_encoder = None
        
        # Fallback: simple text encoder
        if not self.use_sentence_transformers:
            # Character-level embedding
            self.char_embedding = nn.Embedding(128, 64)  # ASCII characters
            self.text_encoder = nn.Sequential(
                nn.LSTM(64, 128, batch_first=True),
                nn.Linear(128, embed_dim)
            )
    
    def forward(
        self,
        text_snippets: List[List[str]],  # [B, N_texts] list of text strings
        text_confidences: Optional[torch.Tensor] = None  # [B, N_texts]
    ) -> Tuple[torch.Tensor, List[List[str]]]:
        """
        Encode OCR text snippets.
        
        Args:
            text_snippets: List of text snippets per image [B, N_texts]
            text_confidences: Optional confidence scores [B, N_texts]
        
        Returns:
            text_embeddings: Text embeddings [B, max_texts, embed_dim]
            valid_texts: Valid text snippets [B, max_texts]
        """
        B = len(text_snippets)
        device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
        
        text_embeddings = []
        valid_texts_list = []
        
        for b in range(B):
            texts = text_snippets[b]
            
            # Limit to max_texts
            texts = texts[:self.max_texts]
            
            # Encode texts
            if self.use_sentence_transformers and self.text_encoder is not None:
                # Use sentence-transformers
                embeddings = self.text_encoder.encode(
                    texts,
                    convert_to_tensor=True,
                    device=device
                )  # [N_texts, embed_dim]
            else:
                # Fallback: character-level encoding
                embeddings = []
                for text in texts:
                    # Convert text to character indices
                    chars = [ord(c) % 128 for c in text[:100]]  # Limit length
                    if not chars:
                        chars = [0]
                    char_tensor = torch.tensor(chars, device=device).unsqueeze(0)
                    char_emb = self.char_embedding(char_tensor)  # [1, len, 64]
                    lstm_out, _ = self.text_encoder[0](char_emb)
                    text_emb = self.text_encoder[1](lstm_out[:, -1, :])  # [1, embed_dim]
                    embeddings.append(text_emb.squeeze(0))
                
                if embeddings:
                    embeddings = torch.stack(embeddings)  # [N_texts, embed_dim]
                else:
                    embeddings = torch.zeros(1, self.embed_dim, device=device)
            
            # Pad to max_texts
            N = embeddings.shape[0]
            if N < self.max_texts:
                padding = torch.zeros(self.max_texts - N, self.embed_dim, device=device)
                embeddings = torch.cat([embeddings, padding], dim=0)
                texts = texts + [''] * (self.max_texts - N)
            
            text_embeddings.append(embeddings[:self.max_texts])
            valid_texts_list.append(texts[:self.max_texts])
        
        text_embeddings = torch.stack(text_embeddings)  # [B, max_texts, embed_dim]
        
        # L2 normalize
        text_embeddings = nn.functional.normalize(text_embeddings, p=2, dim=2)
        
        return text_embeddings, valid_texts_list


