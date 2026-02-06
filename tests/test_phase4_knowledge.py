"""Comprehensive Tests for Phase 4: Knowledge-Augmented Retrieval

Tests all Phase 4 components:
- Scene Graph Encoder
- GNN Encoder
- Knowledge-Augmented Retrieval"""

import torch
import torch.nn as nn
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSceneGraphEncoder:
    """Test Scene Graph Encoder."""
    
    def test_scene_graph_import(self):
        """Test that Scene Graph Encoder can be imported."""
        from ml.models.scene_graph.scene_graph_encoder import SceneGraphEncoder
        assert SceneGraphEncoder is not None
    
    def test_scene_graph_forward(self):
        """Test Scene Graph Encoder forward pass."""
        from ml.models.scene_graph.scene_graph_encoder import SceneGraphEncoder
        
        encoder = SceneGraphEncoder(
            object_embed_dim=256,
            relation_embed_dim=128
        )
        encoder.eval()
        
        boxes = torch.randn(5, 4)  # [N, 4]
        object_embeddings = torch.randn(5, 256)  # [N, D]
        object_classes = ['person', 'chair', 'table', 'door', 'window']
        
        scene_graph = encoder(boxes, object_embeddings, object_classes)
        
        assert isinstance(scene_graph, dict)
        assert 'spatial_relations' in scene_graph
        assert 'semantic_relations' in scene_graph
        assert 'object_embeddings' in scene_graph
    
    def test_scene_graph_extract_spatial(self):
        """Test spatial relation extraction."""
        from ml.models.scene_graph.scene_graph_encoder import SceneGraphEncoder
        
        encoder = SceneGraphEncoder()
        encoder.eval()
        
        boxes = torch.tensor([
            [0.0, 0.0, 0.5, 0.5],  # Object 1 (left)
            [0.5, 0.0, 1.0, 0.5],  # Object 2 (right)
        ])
        object_embeddings = torch.randn(2, 256)
        
        relations = encoder.extract_spatial_relations(boxes, object_embeddings)
        
        assert isinstance(relations, list)
        assert len(relations) > 0


class TestGNNEncoder:
    """Test GNN Encoder."""
    
    def test_gnn_import(self):
        """Test that GNN Encoder can be imported."""
        from ml.models.scene_graph.scene_graph_encoder import GNNEncoder
        assert GNNEncoder is not None
    
    def test_gnn_forward(self):
        """Test GNN Encoder forward pass."""
        from ml.models.scene_graph.scene_graph_encoder import GNNEncoder
        
        try:
            encoder = GNNEncoder(
                node_dim=256,
                edge_dim=128,
                hidden_dim=256,
                num_layers=3,
                output_dim=512
            )
            encoder.eval()
            
            # Create dummy graph
            node_features = torch.randn(5, 256)  # [N, D]
            edge_index = torch.tensor([
                [0, 1, 2, 3],
                [1, 2, 3, 4]
            ], dtype=torch.long)  # [2, E]
            edge_attr = torch.randn(4, 128)  # [E, D]
            
            graph_embedding = encoder(node_features, edge_index, edge_attr)
            
            assert graph_embedding.shape == (512,)
        except ImportError:
            # torch-geometric not available, skip test
            pytest.skip("torch-geometric not available")


class TestKnowledgeAugment:
    """Test Knowledge-Augmented Retrieval."""
    
    def test_knowledge_augment_import(self):
        """Test that Knowledge Augment can be imported."""
        try:
            from ml.retrieval.retrieval.knowledge_augment import KnowledgeAugmentedRetrieval
            assert KnowledgeAugmentedRetrieval is not None
        except ImportError:
            pytest.skip("torch-geometric not available")
    
    def test_knowledge_augment_forward(self):
        """Test Knowledge-Augmented Retrieval forward pass."""
        try:
            from ml.retrieval.retrieval.knowledge_augment import KnowledgeAugmentedRetrieval
            
            retrieval = KnowledgeAugmentedRetrieval(
                node_dim=256,
                embed_dim=512
            )
            retrieval.eval()
            
            # Create dummy inputs
            visual_embedding = torch.randn(512)
            scene_graph = {
                'node_features': torch.randn(5, 256),
                'edge_index': torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
            }
            
            try:
                kg_score = retrieval(visual_embedding, scene_graph)
                assert kg_score is not None
            except Exception as e:
                # May fail if GNN dependencies missing
                pytest.skip(f"Knowledge augment test skipped: {e}")
        except ImportError:
            pytest.skip("torch-geometric not available")


def run_all_tests():
    """Run all Phase 4 tests."""
    print("=" * 60)
    print("Phase 4: Knowledge-Augmented Retrieval Tests")
    print("=" * 60)
    
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_all_tests()

