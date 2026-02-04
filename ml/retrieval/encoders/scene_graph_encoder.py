"""
Scene Graph Encoder for Multi-Vector Retrieval

Encodes scene graphs for retrieval using GNN.
"""

import torch
import torch.nn as nn
from typing import Optional, List
from ml.models.scene_graph.scene_graph_encoder import GNNEncoder


class SceneGraphRetrievalEncoder(nn.Module):
    """
    Scene graph encoder for retrieval.
    
    Uses GNN to encode scene graphs into embeddings.
    """
    
    def __init__(
        self,
        node_dim: int = 256,
        edge_dim: int = 128,
        embed_dim: int = 512,
        num_layers: int = 3
    ):
        super().__init__()
        
        self.gnn_encoder = GNNEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=256,
            num_layers=num_layers,
            output_dim=embed_dim
        )
    
    def forward(
        self,
        node_features: torch.Tensor,  # [N, node_dim]
        edge_index: torch.Tensor,      # [2, E]
        edge_attr: Optional[torch.Tensor] = None  # [E, edge_dim]
    ) -> torch.Tensor:
        """
        Encode scene graph.
        
        Args:
            node_features: Node features [N, node_dim]
            edge_index: Edge indices [2, E]
            edge_attr: Optional edge attributes [E, edge_dim]
        
        Returns:
            Scene graph embedding [embed_dim]
        """
        return self.gnn_encoder(node_features, edge_index, edge_attr)


