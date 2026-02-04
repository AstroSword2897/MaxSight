"""
Comprehensive Tests for Phase 3: Multi-Vector Retrieval System

Tests all Phase 3 components:
- Retrieval Encoders (Global, Region, Patch, Depth, OCR, Audio, Scene Graph)
- Attention-Based Fusion
- FAISS Indexing
- Two-Stage Retrieval
"""

import torch
import torch.nn as nn
import pytest
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestRetrievalEncoders:
    """Test all retrieval encoders."""
    
    def test_global_encoder_import(self):
        """Test that Global Encoder can be imported."""
        from ml.retrieval.encoders.global_encoder import GlobalEncoder
        assert GlobalEncoder is not None
    
    def test_region_encoder_import(self):
        """Test that Region Encoder can be imported."""
        from ml.retrieval.encoders.region_extractor import RegionExtractor
        assert RegionExtractor is not None
    
    def test_patch_encoder_import(self):
        """Test that Patch Extractor can be imported."""
        from ml.retrieval.encoders.patch_extractor import PatchExtractor
        assert PatchExtractor is not None
    
    def test_depth_encoder_import(self):
        """Test that Depth Extractor can be imported."""
        from ml.retrieval.encoders.depth_extractor import DepthExtractor
        assert DepthExtractor is not None
    
    def test_ocr_encoder_import(self):
        """Test that OCR Encoder can be imported."""
        from ml.retrieval.encoders.ocr_encoder import OCREncoder
        assert OCREncoder is not None
    
    def test_audio_encoder_import(self):
        """Test that Audio Encoder can be imported."""
        from ml.retrieval.encoders.audio_encoder import AudioEncoder
        assert AudioEncoder is not None
    
    def test_scene_graph_encoder_import(self):
        """Test that Scene Graph Encoder can be imported."""
        try:
            from ml.retrieval.encoders.scene_graph_encoder import SceneGraphRetrievalEncoder
            assert SceneGraphRetrievalEncoder is not None
        except ImportError:
            pytest.skip("Scene graph encoder dependencies not available")


class TestAttentionFusion:
    """Test Attention-Based Fusion."""
    
    def test_attention_fusion_import(self):
        """Test that Attention Fusion can be imported."""
        from ml.retrieval.fusion.attention_fusion import AttentionFusion
        assert AttentionFusion is not None
    
    def test_attention_fusion_forward(self):
        """Test Attention Fusion forward pass."""
        from ml.retrieval.fusion.attention_fusion import AttentionFusion
        
        fusion = AttentionFusion(
            embedding_dims={'global': 256, 'region': 128, 'patch': 64},
            fused_dim=256
        )
        fusion.eval()
        
        query = torch.randn(2, 256)
        embeddings = {
            'global': torch.randn(2, 256),  # [B, D] not [B, N, D]
            'region': torch.randn(2, 128),
            'patch': torch.randn(2, 64)
        }
        
        fused = fusion(embeddings, query_embedding=query)
        
        assert fused.shape == (2, 256)


class TestFAISSIndexing:
    """Test FAISS Indexing."""
    
    def test_index_builder_import(self):
        """Test that Neural Index Builder can be imported."""
        from ml.retrieval.indexing.neural_index_builder import NeuralIndexBuilder
        assert NeuralIndexBuilder is not None
    
    def test_index_builder_initialization(self):
        """Test Index Builder initialization."""
        from ml.retrieval.indexing.neural_index_builder import NeuralIndexBuilder
        
        builder = NeuralIndexBuilder(
            embed_dim=256,
            index_type='hnsw',
            metric='cosine'
        )
        assert builder is not None
        assert builder.embed_dim == 256
    
    def test_index_building(self):
        """Test building an index."""
        from ml.retrieval.indexing.neural_index_builder import NeuralIndexBuilder
        
        builder = NeuralIndexBuilder(embed_dim=256, index_type='hnsw')
        
        # Create dummy vectors
        vectors = np.random.randn(100, 256).astype('float32')
        
        builder.build_index(vectors)
        
        assert builder.index is not None
    
    def test_index_search(self):
        """Test searching an index."""
        from ml.retrieval.indexing.neural_index_builder import NeuralIndexBuilder
        
        builder = NeuralIndexBuilder(embed_dim=256, index_type='hnsw')
        
        # Build index
        vectors = np.random.randn(100, 256).astype('float32')
        builder.build_index(vectors)
        
        # Search
        query = np.random.randn(1, 256).astype('float32')
        distances, indices = builder.search(query, k=5)
        
        assert len(distances) == 1
        assert len(indices) == 1
        assert len(indices[0]) == 5


class TestTwoStageRetrieval:
    """Test Two-Stage Retrieval."""
    
    def test_stage1_ann_import(self):
        """Test that Stage 1 ANN can be imported."""
        try:
            from ml.retrieval.retrieval.stage1_ann import Stage1ANN
            assert Stage1ANN is not None
        except ImportError:
            pytest.skip("FAISS not available")
    
    def test_stage2_rerank_import(self):
        """Test that Stage 2 Reranking can be imported."""
        from ml.retrieval.retrieval.stage2_rerank import Stage2Reranker
        assert Stage2Reranker is not None
    
    def test_stage1_retrieval(self):
        """Test Stage 1 ANN retrieval."""
        from ml.retrieval.retrieval.stage1_ann import Stage1ANN
        from ml.retrieval.indexing.neural_index_builder import NeuralIndexBuilder
        import numpy as np
        
        # Create a small index for testing
        builder = NeuralIndexBuilder(embed_dim=256, index_type='flat')  # Use flat for small test
        vectors = np.random.randn(100, 256).astype('float32')
        index = builder.build_index(vectors)
        
        retrieval = Stage1ANN(index=index)
        
        # Create dummy query
        query = np.random.randn(256).astype('float32')
        
        distances, indices = retrieval.search(query, k=5)
        
        assert distances is not None
        assert indices is not None
        assert distances.shape[0] == 1  # Batch size
        assert len(indices[0]) == 5  # k results
    
    def test_stage2_reranking(self):
        """Test Stage 2 reranking."""
        from ml.retrieval.retrieval.stage2_rerank import Stage2Reranker
        
        reranker = Stage2Reranker(
            embedding_dims={'global': 256, 'region': 128, 'patch': 64},
            hidden_dim=256
        )
        reranker.eval()
        
        # Create dummy candidates (list of dicts)
        # Use proper dimensions that match embedding_dims
        candidates = [
            {'global': torch.randn(256), 'region': torch.randn(128), 'patch': torch.randn(64)}
            for _ in range(10)
        ]
        
        query_vectors = {
            'global': torch.randn(256),
            'region': torch.randn(128),
            'patch': torch.randn(64)
        }
        
        reranked_scores, reranked_indices = reranker(
            query_vectors, candidates
        )
        
        assert reranked_scores is not None
        assert reranked_indices is not None
        assert len(reranked_scores) == 10
        assert len(reranked_indices) == 10


def run_all_tests():
    """Run all Phase 3 tests."""
    print("=" * 60)
    print("Phase 3: Multi-Vector Retrieval System Tests")
    print("=" * 60)
    
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_all_tests()

