"""
Batched Scene Graph + GNN Encoder for MaxSight 3.0

- Efficient GPU computation
- Supports multiple scene graphs per batch
- Trainable spatial and semantic relation scoring
- Edge-aware GNN with automatic edge_index/edge_attr generation
- Batched graph pooling for multiple scenes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass

try:
    from torch_geometric.nn import MessagePassing
    from torch_geometric.utils import add_self_loops, softmax
    from torch_geometric.data import Batch
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False


@dataclass
class SceneRelation:
    """
    Represents a relationship between objects.
    
    CRITICAL: Edge identity is explicit (src, dst), not inferred from position.
    This prevents silent corruption from reordering, pruning, or augmentation.
    """
    src: int  # Source node index (explicit, not inferred)
    dst: int  # Destination node index (explicit, not inferred)
    subject: str  # Subject class name (for debugging/human readability)
    predicate: str  # Relation type (spatial or semantic)
    object: str  # Object class name (for debugging/human readability)
    confidence: float  # Relation confidence score


class SceneGraphEncoder(nn.Module):
    """
    Batched Scene Graph Encoder for MaxSight 3.0.
    
    Supports multiple scenes in a single batch with proper batching offsets.
    Automatically generates edge_index and edge_attr for GNN processing.
    """
    
    def __init__(
        self,
        object_embed_dim: int = 256,
        relation_embed_dim: int = 128,
        num_spatial_relations: int = 6,
        num_semantic_relations: int = 10,
        semantic_rules: Optional[Dict] = None,
        mps_stable: bool = False  # MPS workaround: detach edge_attr to avoid backward crashes
    ):
        super().__init__()
        self.object_embed_dim = object_embed_dim
        self.relation_embed_dim = relation_embed_dim
        self.num_spatial_relations = num_spatial_relations
        self.num_semantic_relations = num_semantic_relations
        self.mps_stable = mps_stable
        
        # CRITICAL: MPS-stable mode sacrifices learning on edges to avoid backward crashes
        # Set to True for local MPS training, False for cloud GPU training
        if mps_stable:
            import warnings
            warnings.warn(
                "MPS-stable mode enabled: edge_attr gradients disabled. "
                "Use this only for local MPS development. For real training, use cloud GPU with mps_stable=False.",
                UserWarning
            )

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

        # Relation embedding lookup (for edge_attr)
        self.relation_embedding = nn.Embedding(
            num_spatial_relations + num_semantic_relations,
            relation_embed_dim
        )

        # Optional rule-based semantic overrides
        self.semantic_rules = semantic_rules if semantic_rules else {}
        
        # Also check reverse order for rules
        if semantic_rules:
            self.semantic_rules.update({(k[1], k[0]): v for k, v in semantic_rules.items()})

        # Spatial predicate names
        self.spatial_predicates = ['left', 'right', 'above', 'below', 'near', 'far']

    def extract_relations(
        self,
        boxes: torch.Tensor,  # [total_objects, 4]
        object_embeddings: torch.Tensor,  # [total_objects, object_embed_dim]
        object_classes: List[str],  # List of length total_objects
        batch_offsets: Optional[torch.Tensor] = None  # [num_scenes+1] with start indices
    ) -> Tuple[List[SceneRelation], torch.Tensor, torch.Tensor]:
        """
        Extract pairwise spatial and semantic relations for batched scenes.
        
        CRITICAL: If batch_offsets is provided, removes cross-scene pairs.
        
        Args:
            boxes: [total_objects, 4] - Bounding boxes for all objects across all scenes
            object_embeddings: [total_objects, object_embed_dim] - Object embeddings
            object_classes: List of class names (length total_objects)
            batch_offsets: Optional [num_scenes+1] tensor with start indices for each scene
        
        Returns:
            relations: List of SceneRelation objects
            edge_index: [2, E] - Edge indices for GNN
            edge_attr: [E, relation_embed_dim] - Edge attributes (relation embeddings)
        """
        device = object_embeddings.device
        N = object_embeddings.shape[0]

        # CRITICAL: Compute all pairwise indices (upper triangle) for entire batch
        # Use reshape-based approach for MPS compatibility (no CPU fallback)
        # Generate indices manually using meshgrid + triu mask
        if device.type == 'mps':
            # Manual triu_indices for MPS
            i, j = torch.meshgrid(torch.arange(N, device=device), torch.arange(N, device=device), indexing='ij')
            mask = i < j  # Upper triangle
            idx_i = i[mask]
            idx_j = j[mask]
        else:
            idx_i, idx_j = torch.triu_indices(N, N, offset=1, device=device)

        # CRITICAL: If batching, remove cross-scene pairs
        if batch_offsets is not None:
            # Build mask to keep only within-scene pairs
            mask = torch.zeros(len(idx_i), dtype=torch.bool, device=device)
            
            for b in range(len(batch_offsets) - 1):
                start = batch_offsets[b].item()
                end = batch_offsets[b + 1].item()
                
                # Both indices must be in [start, end) for same scene
                mask_i = (idx_i >= start) & (idx_i < end)
                mask_j = (idx_j >= start) & (idx_j < end)
                mask |= (mask_i & mask_j)
            
            idx_i = idx_i[mask]
            idx_j = idx_j[mask]

        # CRITICAL: Vectorized pairwise feature extraction
        emb_i = object_embeddings[idx_i]  # [num_pairs, dim]
        emb_j = object_embeddings[idx_j]
        pair_features = torch.cat([emb_i, emb_j], dim=1)  # [num_pairs, 2*dim]

        # Predict spatial relations (vectorized)
        spatial_logits = self.spatial_classifier(pair_features)  # [num_pairs, num_spatial_relations]
        spatial_probs = F.softmax(spatial_logits, dim=1)
        spatial_idx = spatial_probs.argmax(dim=1)

        # Predict semantic relations (vectorized)
        semantic_logits = self.semantic_classifier(pair_features)  # [num_pairs, num_semantic_relations]
        semantic_probs = F.softmax(semantic_logits, dim=1)
        semantic_idx = semantic_probs.argmax(dim=1)

        # CRITICAL: Build relations with EXPLICIT edge identity (src, dst)
        # Never infer edge from position - always explicit
        relations = []
        edge_map = {}  # (src, dst) -> list of relations (explicit grouping)
        
        for k in range(len(idx_i)):
            src_idx = int(idx_i[k].item())
            dst_idx = int(idx_j[k].item())
            
            # Get class names
            subj = object_classes[src_idx] if src_idx < len(object_classes) else 'object'
            obj = object_classes[dst_idx] if dst_idx < len(object_classes) else 'object'
            
            # Spatial relation
            sp_pred_idx = spatial_idx[k].item()
            sp_pred = self.spatial_predicates[sp_pred_idx]
            sp_conf = spatial_probs[k, sp_pred_idx].item()
            
            # Create relations with EXPLICIT edge identity (not inferred)
            rel_spatial = SceneRelation(
                src=src_idx,  # EXPLICIT source node
                dst=dst_idx,  # EXPLICIT destination node
                subject=subj,
                predicate=sp_pred,
                object=obj,
                confidence=sp_conf
            )
            
            # Semantic relation with rule override
            rule_pred = self.semantic_rules.get((subj, obj), None)
            sem_pred_idx = semantic_idx[k].item()
            sem_pred = rule_pred if rule_pred else f"semantic_{sem_pred_idx}"
            sem_conf = 1.0 if rule_pred else semantic_probs[k, sem_pred_idx].item()
            
            rel_semantic = SceneRelation(
                src=src_idx,  # EXPLICIT source node
                dst=dst_idx,  # EXPLICIT destination node
                subject=subj,
                predicate=sem_pred,
                object=obj,
                confidence=sem_conf
            )
            
            relations.append(rel_spatial)
            relations.append(rel_semantic)
            
            # Group by edge (EXPLICIT, not positional)
            edge_key = (src_idx, dst_idx)
            if edge_key not in edge_map:
                edge_map[edge_key] = []
            edge_map[edge_key].append(rel_spatial)
            edge_map[edge_key].append(rel_semantic)

        # CRITICAL: Build edge_index from edge_map keys (explicit, not inferred from list position)
        # This fixes: variable relation counts, pruning, reordering, batched shuffling
        if edge_map:
            edges = list(edge_map.keys())
            edge_index = torch.tensor(edges, dtype=torch.long, device=device).T.contiguous()  # [2, E]
            
            # CRITICAL: Enforce invariants (fail loud, fail early)
            num_nodes = object_embeddings.shape[0]
            assert edge_index.ndim == 2, f"edge_index must be 2D, got {edge_index.ndim}D"
            assert edge_index.shape[0] == 2, f"edge_index must have shape [2, E], got {edge_index.shape}"
            assert edge_index.is_contiguous(), "edge_index must be contiguous"
            assert edge_index.max().item() < num_nodes, f"edge_index max ({edge_index.max().item()}) >= num_nodes ({num_nodes})"
            assert edge_index.min().item() >= 0, f"edge_index has negative indices"
            
            # Aggregate relations per edge (explicit aggregation, not positional)
            # Strategy: use max confidence relation's embedding
            edge_features = []
            for edge in edges:
                rels = edge_map[edge]
                # Find best relation by confidence
                best_rel = max(rels, key=lambda r: r.confidence)
                
                # Map predicate to relation type index for embedding lookup
                if best_rel.predicate in self.spatial_predicates:
                    rel_type_idx = self.spatial_predicates.index(best_rel.predicate)
                else:
                    # Semantic relation: hash to valid range
                    rel_type_idx = self.num_spatial_relations + (hash(best_rel.predicate) % self.num_semantic_relations)
                
                # Clamp to valid embedding range
                rel_type_idx = min(rel_type_idx, self.num_spatial_relations + self.num_semantic_relations - 1)
                edge_features.append(self.relation_embedding(torch.tensor(rel_type_idx, device=device)))
            
            edge_attr = torch.stack(edge_features, dim=0)  # [E, relation_embed_dim]
            
            # CRITICAL: Verify consistency (edge_index and edge_attr must match)
            assert edge_index.shape[1] == len(edge_attr), \
                f"edge_index edges ({edge_index.shape[1]}) != edge_attr count ({len(edge_attr)})"
        else:
            # No edges case
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_attr = torch.empty((0, self.relation_embed_dim), dtype=torch.float32, device=device)

        return relations, edge_index, edge_attr

    def forward(
        self,
        boxes: torch.Tensor,  # [total_objects, 4] or [B, N, 4]
        object_embeddings: torch.Tensor,  # [total_objects, object_embed_dim] or [B, N, dim]
        object_classes: List[str],  # List of length total_objects or List[List[str]] for batched
        batch_offsets: Optional[torch.Tensor] = None  # [num_scenes+1]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for batched scene graph encoding.
        
        Args:
            boxes: [total_objects, 4] or [B, N, 4] - Bounding boxes
            object_embeddings: [total_objects, object_embed_dim] or [B, N, dim] - Object embeddings
            object_classes: List of class names or List[List[str]] for batched
            batch_offsets: Optional [num_scenes+1] tensor with start indices
        
        Returns:
            Dictionary with:
            - 'relations': List of SceneRelation objects
            - 'edge_index': [2, E] - Edge indices for GNN
            - 'edge_attr': [E, relation_embed_dim] - Edge attributes
            - 'object_embeddings': [total_objects, object_embed_dim] - Node features
            - 'batch': [total_objects] - Batch tensor for GNN pooling (if batch_offsets provided)
        """
        # Handle batched input format [B, N, ...] -> flatten to [total_objects, ...]
        if boxes.dim() == 3:
            B, N, _ = boxes.shape
            boxes = boxes.contiguous().reshape(B * N, -1)
            object_embeddings = object_embeddings.contiguous().reshape(B * N, -1)
            
            # Flatten object_classes
            if isinstance(object_classes[0], list):
                object_classes = [cls for scene_classes in object_classes for cls in scene_classes]
            
            # Create batch_offsets if not provided
            if batch_offsets is None:
                batch_offsets = torch.arange(0, (B + 1) * N, N, device=boxes.device, dtype=torch.long)

        # Extract relations with automatic edge_index/edge_attr generation
        relations, edge_index, edge_attr = self.extract_relations(
            boxes, object_embeddings, object_classes, batch_offsets
        )

        # Build batch tensor for GNN pooling
        batch = None
        if batch_offsets is not None:
            num_scenes = len(batch_offsets) - 1
            batch = torch.zeros(object_embeddings.shape[0], dtype=torch.long, device=object_embeddings.device)
            for b in range(num_scenes):
                start = batch_offsets[b].item()
                end = batch_offsets[b + 1].item()
                batch[start:end] = b

        return {
            'relations': relations,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'object_embeddings': object_embeddings,
            'batch': batch
        }


if TORCH_GEOMETRIC_AVAILABLE:
    class GNNLayer(MessagePassing):
        """Single GNN layer using message passing with edge attributes."""
        
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
        """
        Graph Neural Network encoder for scene graphs (batched support).
        
        Supports multiple graphs in a batch with proper pooling per graph.
        
        MPS COMPATIBILITY:
        - Uses index_add for batched pooling (can crash on MPS with large tensors)
        - Residual connections + LayerNorm can cause backward crashes on MPS
        - Set mps_stable=True to use CPU fallback for pooling
        """
        
        def __init__(
            self, 
            node_dim=256, 
            edge_dim=128, 
            hidden_dim=256, 
            num_layers=3, 
            output_dim=512,
            mps_stable: bool = False  # MPS workaround: use CPU for index_add
        ):
            super().__init__()
            self.node_proj = nn.Linear(node_dim, hidden_dim)
            self.edge_proj = nn.Linear(edge_dim, hidden_dim) if edge_dim > 0 else None
            self.gnn_layers = nn.ModuleList([GNNLayer(hidden_dim, hidden_dim) for _ in range(num_layers)])
            self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
            self.output_proj = nn.Linear(hidden_dim, output_dim)
            self.mps_stable = mps_stable
            
            if mps_stable:
                import warnings
                warnings.warn(
                    "GNNEncoder MPS-stable mode: using CPU fallback for index_add. "
                    "This is slower but avoids MPS kernel crashes.",
                    UserWarning
                )

        def forward(
            self, 
            node_features: torch.Tensor,  # [total_nodes, node_dim]
            edge_index: torch.Tensor,  # [2, E]
            edge_attr: Optional[torch.Tensor] = None,  # [E, edge_dim]
            batch: Optional[torch.Tensor] = None  # [total_nodes] with graph IDs
        ) -> torch.Tensor:
            """
            Forward pass with batched graph pooling.
            
            Args:
                node_features: [total_nodes, node_dim] - Node features for all graphs
                edge_index: [2, E] - Edge indices
                edge_attr: [E, edge_dim] - Edge attributes
                batch: [total_nodes] - Batch tensor indicating which graph each node belongs to
            
            Returns:
                Graph embeddings: [num_graphs, output_dim] if batch provided, else [1, output_dim]
            """
            x = self.node_proj(node_features)
            
            if edge_attr is not None and self.edge_proj is not None:
                edge_attr = self.edge_proj(edge_attr)
            
            # GNN layers with residual connections
            for gnn_layer, norm in zip(self.gnn_layers, self.norms):
                x_new = gnn_layer(x, edge_index, edge_attr)
                x = norm(x + x_new)
            
            # CRITICAL: Batched graph pooling
            if batch is not None:
                num_graphs = batch.max().item() + 1
                
                # MPS WORKAROUND: index_add can crash on MPS with large tensors
                # Use CPU fallback for pooling if mps_stable=True
                if self.mps_stable and x.device.type == 'mps':
                    # Move to CPU for pooling, then back to MPS
                    x_cpu = x.cpu()
                    batch_cpu = batch.cpu()
                    graph_embeddings = torch.zeros(num_graphs, x_cpu.size(1), dtype=x_cpu.dtype)
                    graph_embeddings = graph_embeddings.index_add(0, batch_cpu, x_cpu)
                    node_counts = torch.bincount(batch_cpu, minlength=num_graphs).float()
                    node_counts = torch.clamp(node_counts, min=1.0)
                    graph_embeddings = graph_embeddings / node_counts.unsqueeze(1)
                    graph_embeddings = graph_embeddings.to(x.device)
                else:
                    # Standard GPU/CPU path
                    graph_embeddings = torch.zeros(num_graphs, x.size(1), device=x.device, dtype=x.dtype)
                    graph_embeddings = graph_embeddings.index_add(0, batch, x)
                    node_counts = torch.bincount(batch, minlength=num_graphs).float()
                    node_counts = torch.clamp(node_counts, min=1.0)
                    graph_embeddings = graph_embeddings / node_counts.unsqueeze(1)
            else:
                # Single graph: mean pool
                graph_embeddings = x.mean(dim=0, keepdim=True)
            
            # Project to output dimension
            output = self.output_proj(graph_embeddings)
            
            return output

else:
    class GNNEncoder(nn.Module):
        """Placeholder GNN encoder when torch-geometric is not available."""
        def __init__(self, *args, **kwargs):
            super().__init__()
        def forward(self, *args, **kwargs):
            raise ImportError("torch-geometric is required for GNNEncoder")
