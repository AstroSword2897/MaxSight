"""Tests for optional features that must exist (TimeSformer, MiDaS loader, OCR)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from ml.models.temporal.temporal_transformer import TimeSformer
from ml.retrieval.encoders.depth_extractor import DepthExtractor
from ml.retrieval.encoders.midas_loader import _HubDepthModel, load_model
from ml.retrieval.encoders.ocr_encoder import OCREncoder, SENTENCE_TRANSFORMERS_AVAILABLE


def test_timesformer_import_path():
    model = TimeSformer(embed_dim=64, num_heads=4, num_layers=2, num_frames=4)
    x = torch.randn(2, 4, 8, 64)
    out = model(x)
    assert out.shape == (2, 64)


def test_midas_load_model_uses_torch_hub():
    class FakeHub(nn.Module):
        def forward(self, x):
            return x.mean(dim=1)

    fake = FakeHub()
    with patch("torch.hub.load", return_value=fake) as hub_load:
        model = load_model("DPT_Large")
        hub_load.assert_called_once()
        assert isinstance(model, _HubDepthModel)
        images = torch.randn(2, 3, 32, 32)
        depth = model(images)
        assert depth.shape == (2, 1, 32, 32)


def test_depth_extractor_uses_mocked_midas():
    class FakeDepth(nn.Module):
        def forward(self, images):
            return images.mean(dim=1, keepdim=True)

    extractor = DepthExtractor(embed_dim=32, use_midas=True)
    with patch(
        "ml.retrieval.encoders.depth_extractor.load_model", return_value=FakeDepth()
    ):
        out = extractor(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 32)
    assert torch.allclose(out.norm(dim=1), torch.ones(2), atol=1e-5)


def test_depth_extractor_synthetic_when_disabled():
    extractor = DepthExtractor(embed_dim=16, use_midas=False)
    out = extractor(torch.randn(2, 3, 48, 48))
    assert out.shape == (2, 16)


def test_ocr_encoder_fallback_always_works():
    with patch(
        "ml.retrieval.encoders.ocr_encoder.SENTENCE_TRANSFORMERS_AVAILABLE", False
    ):
        enc = OCREncoder(embed_dim=64, max_texts=3)
        assert enc.use_sentence_transformers is False
        texts = [["hello", "world"], ["a"]]
        emb, valid = enc(texts)
        assert emb.shape == (2, 3, 64)
        assert len(valid) == 2


@pytest.mark.skipif(
    not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not installed"
)
def test_ocr_encoder_sentence_transformers_path():
    mock_st = MagicMock()
    mock_st.encode.return_value = torch.randn(2, 384)

    with patch(
        "ml.retrieval.encoders.ocr_encoder.SentenceTransformer", return_value=mock_st
    ):
        enc = OCREncoder(model_name="all-MiniLM-L6-v2", embed_dim=384, max_texts=4)
        assert enc.use_sentence_transformers is True
        emb, valid = enc([["one", "two"]])
        assert emb.shape == (1, 4, 384)
        assert len(valid[0]) == 4


def test_redis_cache_constructs_with_mocked_client():
    import ml.cache.redis_cache as rc

    fake_client = MagicMock()
    fake_client.setex.return_value = True
    fake_redis = MagicMock()
    fake_redis.from_url.return_value = fake_client
    with (
        patch.object(rc, "REDIS_AVAILABLE", True),
        patch.object(rc, "redis", fake_redis, create=True),
    ):
        cache = rc.RedisCache(redis_url="redis://localhost:6379/0")
        assert cache.set("k", {"a": 1}, ttl=10) is True
        fake_client.setex.assert_called_once()
        fake_client.get.return_value = None
        assert cache.get("missing") is None
