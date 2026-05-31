"""Concept-Dimensioned Retrieval for MaxSight 3.0."""

import torch
import torch.nn as nn


class ConceptRetrieval(nn.Module):
    """Concept-dimensioned retrieval with query-adaptive weights."""

    def __init__(self, num_concepts: int = 10, embed_dim: int = 512):
        super().__init__()
        self.num_concepts = num_concepts
        self.concept_names = [
            "geometry",
            "objects",
            "text",
            "texture",
            "color",
            "vegetation",
            "urban",
            "material",
            "lighting",
            "context",
        ]
        self.concept_predictor = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.ReLU(), nn.Linear(256, num_concepts), nn.Softmax(dim=1)
        )

    def forward(self, query_embedding: torch.Tensor) -> torch.Tensor:
        """Predict concept weights for query."""
        return self.concept_predictor(query_embedding)
