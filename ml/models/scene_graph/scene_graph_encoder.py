"""
Batched Scene Graph + GNN Encoder for MaxSight 3.0

- Efficient GPU computation
- Supports multiple scene graphs per batch
- Trainable spatial and semantic relation scoring
- Edge-aware GNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict
from dataclasses import dataclass

try:
    from torch_geometric.nn import MessagePassing
    from torch_geometric.utils import add_self_loops, softmax
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False


@dataclass
class SceneRelation:
    subject: str
    predicate: str
    object: str
    confidence: float
    src: int = 0  # Source node index
    dst: int = 0  # Destination node index


class SceneGraphEncoder(nn.Module):
    def __init__(
        self,
        object_embed_dim: int = 256,
        relation_embed_dim: int = 128,
        num_spatial_relations: int = 6,
        num_semantic_relations: int = 10,
        semantic_rules: Optional[Dict] = None,
        mps_stable: bool = False
    ):
        super().__init__()
        self.object_embed_dim = object_embed_dim
        self.relation_embed_dim = relation_embed_dim
        self.num_spatial_relations = num_spatial_relations
        self.num_semantic_relations = num_semantic_relations
        self.mps_stable = mps_stable

        # Trainable spatial classifier
        self.spatial_classifier = nn.Sequential(
            nn.Linear(object_embed_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, num_spatial_relations)
        )

        # Trainable semantic classifier
        self.semantic_classifier = nn.Sequential(
            nn.Linear(object_embed_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, num_semantic_relations)
        )

        # Relation embeddings
        self.relation_embedding = nn.Embedding(
            num_spatial_relations + num_semantic_relations,
            relation_embed_dim
        )

        # Optional rule-based semantic overrides
        self.semantic_rules = semantic_rules if semantic_rules else {}

        # Predicates
        self.spatial_predicates = ['left', 'right', 'above', 'below', 'near', 'far']

    def extract_spatial_relations(
        self,
        boxes: torch.Tensor,          # [N, 4]
        object_embeddings: torch.Tensor  # [N, object_embed_dim]
    ) -> Dict[str, torch.Tensor]:
        N = boxes.shape[0]
        device = boxes.device

        # Pairwise feature concatenation
        idx_i, idx_j = torch.triu_indices(N, N, offset=1, device=device)
        emb_i = object_embeddings[idx_i]  # [num_pairs, dim]
        emb_j = object_embeddings[idx_j]
        pair_features = torch.cat([emb_i, emb_j], dim=1)  # [num_pairs, 2*dim]

        # Predict spatial relations
        logits = self.spatial_classifier(pair_features)  # [num_pairs, num_relations]
        probs = F.softmax(logits, dim=1)
        pred_idx = probs.argmax(dim=1)

        relations = []
        for k in range(pred_idx.shape[0]):
            relations.append(
                SceneRelation(
                    subject=f"object_{idx_i[k].item()}",
                    predicate=self.spatial_predicates[pred_idx[k].item()],
                    object=f"object_{idx_j[k].item()}",
                    confidence=probs[k, pred_idx[k]].item(),
                    src=idx_i[k].item(),
                    dst=idx_j[k].item()
                )
            )

        return relations

    def extract_semantic_relations(
        self,
        object_classes: List[str],
        object_embeddings: torch.Tensor
    ) -> List[SceneRelation]:
        N = len(object_classes)
        device = object_embeddings.device

        # Pairwise embeddings
        idx_i, idx_j = torch.triu_indices(N, N, offset=1, device=device)
        emb_i = object_embeddings[idx_i]
        emb_j = object_embeddings[idx_j]
        pair_features = torch.cat([emb_i, emb_j], dim=1)

        # Predict semantic relations
        logits = self.semantic_classifier(pair_features)  # [num_pairs, num_semantic_relations]
        probs = F.softmax(logits, dim=1)
        pred_idx = probs.argmax(dim=1)

        relations = []
        for k in range(pred_idx.shape[0]):
            subj = object_classes[idx_i[k].item()]
            obj = object_classes[idx_j[k].item()]
            # Check rule override
            rule_pred = self.semantic_rules.get((subj, obj), None)
            predicate = rule_pred if rule_pred else f"semantic_{pred_idx[k].item()}"
            confidence = 1.0 if rule_pred else probs[k, pred_idx[k]].item()
            relations.append(SceneRelation(
                subject=subj,
                predicate=predicate,
                object=obj,
                confidence=confidence,
                src=idx_i[k].item(),
                dst=idx_j[k].item()
            ))

        return relations

    def forward(
        self,
        boxes: torch.Tensor,
        object_embeddings: torch.Tensor,
        object_classes: List[str]
    ) -> Dict[str, object]:
        # Handle batched input: [B, K, 4] boxes, [B, K, C] embeddings, List[List[str]] classes
        if boxes.dim() == 3:
            # Batched input - process each scene separately
            batch_size = boxes.shape[0]
            all_spatial_relations = []
            all_semantic_relations = []
            all_relations = []
            
            for b in range(batch_size):
                scene_boxes = boxes[b]  # [K, 4]
                scene_embeddings = object_embeddings[b]  # [K, C]
                scene_classes = object_classes[b] if isinstance(object_classes[0], list) else object_classes
                
                spatial_rels = self.extract_spatial_relations(scene_boxes, scene_embeddings)
                semantic_rels = self.extract_semantic_relations(scene_classes, scene_embeddings)
                
                all_spatial_relations.extend(spatial_rels)
                all_semantic_relations.extend(semantic_rels)
                all_relations.extend(spatial_rels + semantic_rels)
            
            # Build edge_index and edge_attr for batched graphs
            # For now, return simple structure - can be enhanced later
            edge_index = torch.empty((2, 0), dtype=torch.long, device=object_embeddings.device)
            edge_attr = torch.empty((0, self.relation_embed_dim), dtype=torch.float32, device=object_embeddings.device)
            batch = torch.arange(batch_size, device=object_embeddings.device).repeat_interleave(boxes.shape[1])
            
            return {
                'relations': all_relations,
                'spatial_relations': all_spatial_relations,
                'semantic_relations': all_semantic_relations,
                'all_relations': all_relations,
                'edge_index': edge_index,
                'edge_attr': edge_attr,
                'object_embeddings': object_embeddings,
                'batch': batch
            }
        else:
            # Single scene input
            spatial_relations = self.extract_spatial_relations(boxes, object_embeddings)
            semantic_relations = self.extract_semantic_relations(object_classes, object_embeddings)
            all_relations = spatial_relations + semantic_relations
            
            # Build edge_index and edge_attr for single scene
            edge_index = torch.empty((2, 0), dtype=torch.long, device=object_embeddings.device)
            edge_attr = torch.empty((0, self.relation_embed_dim), dtype=torch.float32, device=object_embeddings.device)
            batch = torch.zeros(boxes.shape[0], dtype=torch.long, device=object_embeddings.device)
            
            return {
                'relations': all_relations,
                'spatial_relations': spatial_relations,
                'semantic_relations': semantic_relations,
                'all_relations': all_relations,
                'edge_index': edge_index,
                'edge_attr': edge_attr,
                'object_embeddings': object_embeddings,
                'batch': batch
            }


if TORCH_GEOMETRIC_AVAILABLE:
    class GNNLayer(MessagePassing):
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__(aggr='add')
            self.lin = nn.Linear(in_channels, out_channels)

        def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None):
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
            return self.propagate(edge_index, x=x, edge_attr=edge_attr)

        def message(self, x_j, edge_attr=None):
            if edge_attr is not None:
                return x_j + edge_attr
            return x_j

        def update(self, aggr_out):
            return self.lin(aggr_out)

    class GNNEncoder(nn.Module):
        def __init__(self, node_dim=256, edge_dim=128, hidden_dim=256, num_layers=3, output_dim=512, mps_stable: bool = False):
            super().__init__()
            self.node_proj = nn.Linear(node_dim, hidden_dim)
            self.edge_proj = nn.Linear(edge_dim, hidden_dim) if edge_dim > 0 else None
            self.gnn_layers = nn.ModuleList([GNNLayer(hidden_dim, hidden_dim) for _ in range(num_layers)])
            self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
            self.output_proj = nn.Linear(hidden_dim, output_dim)
            self.mps_stable = mps_stable

        def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None):
            x = self.node_proj(node_features)
            if edge_attr is not None and self.edge_proj is not None:
                edge_attr = self.edge_proj(edge_attr)
            for gnn_layer, norm in zip(self.gnn_layers, self.norms):
                x_new = gnn_layer(x, edge_index, edge_attr)
                x = norm(x + x_new)
            graph_embedding = x.mean(dim=0)
            output = self.output_proj(graph_embedding)
            return output

else:
    class GNNEncoder(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
        def forward(self, *args, **kwargs):
            raise ImportError("torch-geometric is required for GNNEncoder")

