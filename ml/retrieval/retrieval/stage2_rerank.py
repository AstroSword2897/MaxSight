"""Stage 2: Multi-Vector Reranking for MaxSight 3.0."""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional


class Stage2Reranker(nn.Module):
    """Reranks candidates using multi-vector weighted scoring."""
    
    def __init__(self, embedding_dims: Dict[str, int], hidden_dim: int = 256, num_concepts: int = 10):
        super().__init__()
        self.embedding_dims = embedding_dims
        self.num_concepts = num_concepts
        self.concept_weights = nn.Parameter(torch.randn(num_concepts, len(embedding_dims)) * 0.02)
        self.rerank_mlp = nn.Sequential(
            nn.Linear(sum(embedding_dims.values()), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, query_embeddings: Dict[str, torch.Tensor], 
                candidate_embeddings: List[Dict[str, torch.Tensor]],
                concept_weights: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        N = len(candidate_embeddings)
        device = next(iter(query_embeddings.values())).device
        scores = []
        for candidate in candidate_embeddings:
            similarities = []
            for name in query_embeddings.keys():
                if name in candidate:
                    q_emb = query_embeddings[name]
                    c_emb = candidate[name]
                    if q_emb.dim() == 1:
                        q_emb = q_emb.unsqueeze(0)
                    if c_emb.dim() == 1:
                        c_emb = c_emb.unsqueeze(0)
                    q_emb = q_emb.mean(dim=0) if q_emb.dim() > 1 else q_emb.squeeze(0)
                    c_emb = c_emb.mean(dim=0) if c_emb.dim() > 1 else c_emb.squeeze(0)
                    sim = torch.dot(q_emb, c_emb) / (torch.norm(q_emb) * torch.norm(c_emb) + 1e-8)
                    similarities.append(sim)
                else:
                    similarities.append(torch.tensor(0.0, device=device))
            if concept_weights is not None:
                weighted_sim = torch.sum(concept_weights.unsqueeze(1) * self.concept_weights * torch.stack(similarities), dim=0).sum()
            else:
                weighted_sim = torch.stack(similarities).mean()
            # Aggregate query embeddings properly (handle multi-dimensional tensors)
            query_vecs = []
            for name in self.embedding_dims.keys():
                if name in query_embeddings:
                    q_emb = query_embeddings[name]
                    # If multi-dimensional, take mean or flatten appropriately.
                    if q_emb.dim() > 1:
                        # For region/patch embeddings, take mean across spatial dimensions.
                        q_emb = q_emb.mean(dim=0) if q_emb.dim() > 1 else q_emb
                    # Ensure it matches expected dimension.
                    if q_emb.numel() != self.embedding_dims[name]:
                        # Reshape or project to expected dimension.
                        if q_emb.numel() > self.embedding_dims[name]:
                            q_emb = q_emb.flatten()[:self.embedding_dims[name]]
                        else:
                            q_emb = torch.cat([q_emb.flatten(), torch.zeros(self.embedding_dims[name] - q_emb.numel(), device=device)])
                    query_vecs.append(q_emb.flatten()[:self.embedding_dims[name]])
                else:
                    query_vecs.append(torch.zeros(self.embedding_dims[name], device=device))
            combined = torch.cat(query_vecs)
            mlp_score = self.rerank_mlp(combined).squeeze()
            final_score = weighted_sim + 0.1 * mlp_score
            scores.append(final_score)
        scores = torch.stack(scores)
        sorted_indices = torch.argsort(scores, descending=True)
        return scores[sorted_indices], sorted_indices






