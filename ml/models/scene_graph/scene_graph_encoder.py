"""
Scene Graph and GNN Encoders for MaxSight 3.0

Extracts object relationships, builds semantic scene graphs, and encodes them with GNN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
try:
    from torch_geometric.nn import MessagePassing
    from torch_geometric.utils import add_self_loops, degree
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False


@dataclass
class SceneRelation:
    """Represents a relationship between objects."""
    subject: str
    predicate: str
    object: str
    confidence: float


class SceneGraphEncoder(nn.Module):
    """
    Scene graph encoder.
    
    Extracts object relationships:
    - Spatial relationships (left, right, above, below, near, far)
    - Semantic relationships (part-of, contains, supports, etc.)
    - Generates scene graph embeddings
    """
    
    def __init__(
        self,
        object_embed_dim: int = 256,
        relation_embed_dim: int = 128,
        num_spatial_relations: int = 6,  # left, right, above, below, near, far
        num_semantic_relations: int = 10  # part-of, contains, supports, etc.
    ):
        super().__init__()
        
        self.object_embed_dim = object_embed_dim
        self.relation_embed_dim = relation_embed_dim
        
        # Spatial relation classifier
        self.spatial_classifier = nn.Sequential(
            nn.Linear(object_embed_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, num_spatial_relations),
            nn.Softmax(dim=1)
        )
        
        # Semantic relation classifier
        self.semantic_classifier = nn.Sequential(
            nn.Linear(object_embed_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, num_semantic_relations),
            nn.Softmax(dim=1)
        )
        
        # Relation embedding
        self.relation_embedding = nn.Embedding(
            num_spatial_relations + num_semantic_relations,
            relation_embed_dim
        )
    
    def extract_spatial_relations(
        self,
        boxes: torch.Tensor,  # [N, 4] (x1, y1, x2, y2)
        object_embeddings: torch.Tensor  # [N, object_embed_dim]
    ) -> List[SceneRelation]:
        """
        Extract spatial relationships between objects.
        
        Args:
            boxes: Bounding boxes [N, 4]
            object_embeddings: Object embeddings [N, object_embed_dim]
        
        Returns:
            List of spatial relations
        """
        N = boxes.shape[0]
        relations = []
        
        spatial_predicates = ['left', 'right', 'above', 'below', 'near', 'far']
        
        for i in range(N):
            for j in range(i + 1, N):
                box_i = boxes[i]
                box_j = boxes[j]
                
                # Compute spatial features
                center_i = torch.tensor([
                    (box_i[0] + box_i[2]) / 2,
                    (box_i[1] + box_i[3]) / 2
                ])
                center_j = torch.tensor([
                    (box_j[0] + box_j[2]) / 2,
                    (box_j[1] + box_j[3]) / 2
                ])
                
                # Combine embeddings
                combined = torch.cat([object_embeddings[i], object_embeddings[j]])
                
                # Classify spatial relation
                spatial_logits = self.spatial_classifier(combined.unsqueeze(0))
                pred_idx = spatial_logits.argmax().item()
                confidence = spatial_logits[0, pred_idx].item()
                
                relations.append(SceneRelation(
                    subject=f"object_{i}",
                    predicate=spatial_predicates[pred_idx],
                    object=f"object_{j}",
                    confidence=confidence
                ))
        
        return relations
    
    def extract_semantic_relations(
        self,
        object_classes: List[str],
        object_embeddings: torch.Tensor  # [N, object_embed_dim]
    ) -> List[SceneRelation]:
        """
        Extract semantic relationships based on object classes.
        
        Args:
            object_classes: List of object class names
            object_embeddings: Object embeddings [N, object_embed_dim]
        
        Returns:
            List of semantic relations
        """
        N = len(object_classes)
        relations = []
        
        # Semantic relationship rules (can be learned)
        semantic_rules = {
            ('door', 'door_handle'): 'has-part',
            ('chair', 'table'): 'near',
            ('person', 'chair'): 'sits-on',
            # Add more rules
        }
        
        for i in range(N):
            for j in range(i + 1, N):
                class_i = object_classes[i]
                class_j = object_classes[j]
                
                # Check semantic rules
                if (class_i, class_j) in semantic_rules:
                    predicate = semantic_rules[(class_i, class_j)]
                    relations.append(SceneRelation(
                        subject=class_i,
                        predicate=predicate,
                        object=class_j,
                        confidence=1.0
                    ))
        
        return relations
    
    def forward(
        self,
        boxes: torch.Tensor,  # [N, 4]
        object_embeddings: torch.Tensor,  # [N, object_embed_dim]
        object_classes: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Extract scene graph from detections.
        
        Args:
            boxes: Bounding boxes [N, 4]
            object_embeddings: Object embeddings [N, object_embed_dim]
            object_classes: List of class names
        
        Returns:
            Dictionary with scene graph components
        """
        # Extract spatial relations
        spatial_relations = self.extract_spatial_relations(boxes, object_embeddings)
        
        # Extract semantic relations
        semantic_relations = self.extract_semantic_relations(object_classes, object_embeddings)
        
        # Encode relations
        all_relations = spatial_relations + semantic_relations
        
        # Return scene graph representation
        return {
            'spatial_relations': spatial_relations,
            'semantic_relations': semantic_relations,
            'all_relations': all_relations,
            'object_embeddings': object_embeddings
        }


if TORCH_GEOMETRIC_AVAILABLE:
    class GNNLayer(MessagePassing):
        """Single GNN layer using message passing."""
        
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__(aggr='add')
            self.lin = nn.Linear(in_channels, out_channels)
        
        def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
            return self.propagate(edge_index, x=x)
        
        def message(self, x_j: torch.Tensor) -> torch.Tensor:
            return x_j
        
        def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
            return self.lin(aggr_out)
    
    
    class GNNEncoder(nn.Module):
        """Graph Neural Network encoder for scene graphs."""
        
        def __init__(self, node_dim: int = 256, edge_dim: int = 128, hidden_dim: int = 256, 
                     num_layers: int = 3, output_dim: int = 512):
            super().__init__()
            self.node_dim = node_dim
            self.edge_dim = edge_dim
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.node_proj = nn.Linear(node_dim, hidden_dim)
            self.edge_proj = nn.Linear(edge_dim, hidden_dim) if edge_dim > 0 else None
            self.gnn_layers = nn.ModuleList([GNNLayer(hidden_dim, hidden_dim) for _ in range(num_layers)])
            self.output_proj = nn.Linear(hidden_dim, output_dim)
            self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        
        def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor, 
                   edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
            x = self.node_proj(node_features)
            for gnn_layer, norm in zip(self.gnn_layers, self.norms):
                x_new = gnn_layer(x, edge_index)
                x = norm(x + x_new)
            graph_embedding = x.mean(dim=0)
            output = self.output_proj(graph_embedding)
            return output
else:
    class GNNEncoder(nn.Module):
        """Placeholder GNN encoder when torch-geometric is not available."""
        def __init__(self, *args, **kwargs):
            super().__init__()
        def forward(self, *args, **kwargs):
            raise ImportError("torch-geometric is required for GNNEncoder")


