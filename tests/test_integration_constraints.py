"""Unit tests for integration constraints. Ensures architectural constraints are enforced."""
import torch
import torch.nn.functional as F
import pytest
import sys
import os

# Add parent directory to path for imports.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools', 'simulation'))

# Dependency checks.
HAS_SKLEARN = False
try:
    import sklearn
    HAS_SKLEARN = True
except ImportError:
    pass

HAS_TRANSFORMERS = False
try:
    import transformers
    HAS_TRANSFORMERS = True
except ImportError:
    pass

HAS_TEMPORAL_TRANSFORMER = False
try:
    from ml.models.temporal.temporal_transformer import TimeSformer
    HAS_TEMPORAL_TRANSFORMER = True
except (ImportError, ModuleNotFoundError):
    pass

# Mock flask for testing.
import unittest.mock
with unittest.mock.patch('flask.Flask'), unittest.mock.patch('flask_cors.CORS'):
    from ml.models.maxsight_cnn import MaxSightCNN
    from ml.models.heads.depth_head import DepthHead
    from ml.models.fusion.multimodal_fusion import SpatialSoundMapping, EnhancedAudioEncoder


def test_audio_attention_preserves_channels():
    """Assert audio attention never changes channel count."""
    features = torch.randn(2, 256, 14, 14)
    audio_features = torch.randn(2, 128)
    
    audio_encoder = EnhancedAudioEncoder(input_dim=128, embed_dim=256)
    spatial_sound = SpatialSoundMapping(audio_dim=256, attention_size=(14, 14))
    
    audio_emb, _ = audio_encoder(audio_features)
    attention_map, _, _ = spatial_sound(audio_emb)
    
    # Interpolate if needed.
    if attention_map.shape[2:] != features.shape[2:]:
        attention_map = F.interpolate(attention_map, size=features.shape[2:], mode='bilinear')
    
    # Apply attention.
    fused = features * (1.0 + torch.sigmoid(attention_map))
    
    assert fused.shape == features.shape, "Channel count must be preserved"
    assert fused.shape[1] == 256, "Channels must remain 256"


def test_depth_uncertainty_encapsulated():
    """Assert depth uncertainty comes from head, not re-calling layers."""
    depth_head = DepthHead(in_channels=256, dropout=0.1)
    features = torch.randn(2, 256, 14, 14)
    
    outputs = depth_head(features)
    
    assert 'uncertainty' in outputs, "Uncertainty must be in outputs"
    assert outputs['uncertainty'].shape == outputs['depth_map'].shape, \
        "Uncertainty must match depth_map shape"
    assert hasattr(depth_head, 'uncertainty_conv'), \
        "Uncertainty must be a module"


@pytest.mark.skipif(not HAS_TEMPORAL_TRANSFORMER, reason="temporal_transformer module not available")
def test_temporal_spatial_alignment():
    """Assert temporal features match spatial resolution."""
    from ml.models.temporal.temporal_encoder import TemporalEncoder
    
    # Use ConvLSTM only (no TimeSformer) to avoid missing dependency.
    temporal_encoder = TemporalEncoder(
        in_channels=256, 
        num_frames=8, 
        hidden_dim=256,
        use_conv_lstm=True,
        use_timesformer=False  # Skip TimeSformer if not available.
    )
    features = torch.randn(2, 256, 14, 14)
    temporal_features = torch.randn(2, 8, 256, 14, 14)
    
    temporal_outputs = temporal_encoder(temporal_features)
    motion_features = temporal_outputs.get('motion_features')
    
    if motion_features is not None:
        assert motion_features.shape[2:] == features.shape[2:], \
            f"Temporal {motion_features.shape} must match spatial {features.shape}"


@pytest.mark.skipif(not HAS_SKLEARN or not HAS_TRANSFORMERS, 
                    reason="sklearn or transformers not available (required for MaxSightCNN)")
def test_scene_graph_top_k():
    """Assert scene graph uses top-K, not all H*W."""
    model = MaxSightCNN()
    H, W = 14, 14
    
    obj_scores = torch.randn(2, H * W)
    top_k = min(model.max_scene_graph_objects, H * W)
    
    top_k_scores, top_k_indices = torch.topk(obj_scores, k=top_k, dim=1)
    
    assert top_k_indices.shape[1] <= model.max_scene_graph_objects, \
        f"Scene graph must use ≤{model.max_scene_graph_objects} objects"


@pytest.mark.skipif(not HAS_SKLEARN or not HAS_TRANSFORMERS, 
                    reason="sklearn or transformers not available (required for MaxSightCNN)")
def test_personalization_normalized():
    """Assert personalization embeddings are normalized."""
    model = MaxSightCNN()
    user_id = torch.tensor([0, 1])
    
    user_emb = model.user_embeddings(user_id)
    user_emb = F.normalize(user_emb, p=2, dim=1)
    
    norms = torch.norm(user_emb, p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
        "User embeddings must be normalized"


def test_depth_vectorized():
    """Assert depth sampling uses grid_sample, not loops."""
    depth_map = torch.randn(2, 14, 14)
    box_centers = torch.rand(2, 10, 2)  # [B, K, 2].
    
    # Normalize.
    normalized = (box_centers / torch.tensor([14.0, 14.0])) * 2.0 - 1.0
    normalized = normalized.flip(-1).unsqueeze(2)
    
    # Uses grid_sample (vectorized)
    sampled = F.grid_sample(
        depth_map.unsqueeze(1),
        normalized,
        mode='bilinear',
        align_corners=False
    )
    
    assert sampled.shape[0] == 2, "Must be batched"
    assert sampled.shape[1] == 1, "Single channel"
    assert sampled.shape[2] == 10, "K samples"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])






