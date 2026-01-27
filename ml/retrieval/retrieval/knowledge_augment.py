"""Knowledge-Augmented Retrieval with GNN for MaxSight 3.0."""

import torch
import torch.nn as nn
from typing import Dict, Optional, List
from ml.models.scene_graph.scene_graph_encoder import GNNEncoder


class KnowledgeAugmentedRetrieval(nn.Module):
    """GNN-enhanced knowledge-augmented retrieval."""
    
    def __init__(self, node_dim: int = 256, embed_dim: int = 512):
        super().__init__()
        self.gnn_encoder = GNNEncoder(node_dim, edge_dim=128, hidden_dim=256, num_layers=3, output_dim=embed_dim)
        self.knowledge_scorer = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, visual_embedding: torch.Tensor, scene_graph: Dict) -> torch.Tensor:
        """Combine visual retrieval score with KG score."""
        node_features = scene_graph['node_features']
        edge_index = scene_graph['edge_index']
        kg_embedding = self.gnn_encoder(node_features, edge_index)
        combined = torch.cat([visual_embedding, kg_embedding], dim=0)
        kg_score = self.knowledge_scorer(combined.unsqueeze(0)).squeeze()
        return kg_score

