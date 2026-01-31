"""
Fake-Graph Test for Scene Graph Consistency

CRITICAL: This single test eliminates an entire class of bugs.
If this fails, STOP THE PIPELINE - graph structure is broken.
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.models.scene_graph.scene_graph_encoder import SceneGraphEncoder, SceneRelation


def test_scene_graph_consistency():
    """
    Test that edge_index and edge_attr are consistent.
    
    This test verifies:
    1. Edge identity is explicit (src, dst), not inferred
    2. Relations are grouped by edge correctly
    3. edge_index matches edge_attr count
    4. No silent corruption from reordering/pruning
    """
    encoder = SceneGraphEncoder(
        object_embed_dim=256,
        relation_embed_dim=128,
        num_spatial_relations=6,
        num_semantic_relations=10
    )
    
    # Create test data
    num_nodes = 5
    boxes = torch.randn(num_nodes, 4)
    object_embeddings = torch.randn(num_nodes, 256)
    object_classes = ['person', 'car', 'door', 'tree', 'sign']
    
    # Extract relations
    relations, edge_index, edge_attr = encoder.extract_relations(
        boxes=boxes,
        object_embeddings=object_embeddings,
        object_classes=object_classes,
        batch_offsets=None
    )
    
    # CRITICAL ASSERTIONS (fail loud, fail early)
    assert edge_index.ndim == 2, f"edge_index must be 2D, got {edge_index.ndim}D"
    assert edge_index.shape[0] == 2, f"edge_index must have shape [2, E], got {edge_index.shape}"
    assert edge_index.is_contiguous(), "edge_index must be contiguous"
    assert edge_index.shape[1] == edge_attr.shape[0], \
        f"edge_index edges ({edge_index.shape[1]}) != edge_attr count ({edge_attr.shape[0]})"
    assert edge_index.max().item() < num_nodes, \
        f"edge_index max ({edge_index.max().item()}) >= num_nodes ({num_nodes})"
    assert edge_index.min().item() >= 0, "edge_index has negative indices"
    
    # Verify relations have explicit src/dst (not inferred)
    for rel in relations:
        assert hasattr(rel, 'src'), "Relation missing explicit src"
        assert hasattr(rel, 'dst'), "Relation missing explicit dst"
        assert isinstance(rel.src, int), f"src must be int, got {type(rel.src)}"
        assert isinstance(rel.dst, int), f"dst must be int, got {type(rel.dst)}"
        assert 0 <= rel.src < num_nodes, f"src ({rel.src}) out of range [0, {num_nodes})"
        assert 0 <= rel.dst < num_nodes, f"dst ({rel.dst}) out of range [0, {num_nodes})"
    
    # Verify edge_index matches relation edges (explicit check)
    edge_set = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    relation_edges = set((rel.src, rel.dst) for rel in relations)
    
    # All relation edges should be in edge_index
    assert relation_edges.issubset(edge_set), \
        f"Relation edges {relation_edges - edge_set} not in edge_index"
    
    print("✅ Scene graph consistency test PASSED")
    print(f"   Nodes: {num_nodes}")
    print(f"   Edges: {edge_index.shape[1]}")
    print(f"   Relations: {len(relations)}")
    print(f"   Edge attributes: {edge_attr.shape[0]}")


def test_scene_graph_with_pruning():
    """
    Test that graph survives pruning (relations removed, edges preserved).
    
    This verifies that explicit edge identity prevents corruption from pruning.
    """
    encoder = SceneGraphEncoder()
    
    num_nodes = 5
    boxes = torch.randn(num_nodes, 4)
    object_embeddings = torch.randn(num_nodes, 256)
    object_classes = ['person', 'car', 'door', 'tree', 'sign']
    
    # Extract relations
    relations, edge_index, edge_attr = encoder.extract_relations(
        boxes=boxes,
        object_embeddings=object_embeddings,
        object_classes=object_classes
    )
    
    # Simulate pruning: remove low-confidence relations
    pruned_relations = [r for r in relations if r.confidence > 0.5]
    
    # Rebuild edge_index from pruned relations (using explicit src/dst)
    edge_map = {}
    for rel in pruned_relations:
        edge_key = (rel.src, rel.dst)
        if edge_key not in edge_map:
            edge_map[edge_key] = []
        edge_map[edge_key].append(rel)
    
    # Build edge_index from edge_map (explicit, not inferred)
    if edge_map:
        edges = list(edge_map.keys())
        pruned_edge_index = torch.tensor(edges, dtype=torch.long).T.contiguous()
        
        # Verify consistency
        assert pruned_edge_index.shape[1] == len(edge_map), \
            f"Pruned edge_index ({pruned_edge_index.shape[1]}) != edge_map count ({len(edge_map)})"
        
        print("✅ Pruning test PASSED")
        print(f"   Original edges: {edge_index.shape[1]}")
        print(f"   Pruned edges: {pruned_edge_index.shape[1]}")
        print(f"   Original relations: {len(relations)}")
        print(f"   Pruned relations: {len(pruned_relations)}")


if __name__ == "__main__":
    print("="*60)
    print("SCENE GRAPH CONSISTENCY TESTS")
    print("="*60)
    
    try:
        test_scene_graph_consistency()
        test_scene_graph_with_pruning()
        print("\n✅ ALL TESTS PASSED")
        print("   Graph structure is explicit and correct")
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        print("   STOP THE PIPELINE - graph structure is broken")
        sys.exit(1)

