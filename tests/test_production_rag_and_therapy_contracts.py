import sys
from pathlib import Path

import torch

# Ensure project root is on sys.path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_rag_retrieval_is_advisory_only_in_forward_outputs() -> None:
    """
    Retrieval may run asynchronously, but forward must not expose retrieval outputs
    as part of the runtime tensor surface (advisory-only contract).
    """
    from ml.models.maxsight_cnn import CapabilityTier, TierConfig, create_model

    tier_config = TierConfig.for_tier(CapabilityTier.T5_TEMPORAL)
    tier_config.use_retrieval = True
    tier_config.mvp_runtime = True

    model = create_model(use_audio=False, tier_config=tier_config)
    model.eval()

    dummy_image = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        outputs = model(dummy_image)

    # The forward path currently computes retrieval_results but does not condition
    # decoder outputs on it, and does not export retrieval results.
    assert "retrieval_results" not in outputs
    assert "retrieval" not in outputs
    assert "distances" not in outputs
    assert "indices" not in outputs


def test_rag_async_retrieve_non_blocking_contract() -> None:
    """
    Non-blocking contract at the retrieval layer:
    - `blocking=False` must return immediately (return value may be `None`).
    - Retrieval must not raise even if stage1 ANN is not initialized yet.
    """
    from ml.retrieval.retrieval.async_retrieval import AsyncRetrievalSystem

    system = AsyncRetrievalSystem(stage1_ann=None, enable_async=True)
    query_embeddings = {"global": torch.zeros(1, 512).cpu().numpy()}

    result = system.retrieve(query_embeddings=query_embeddings, request_id="t1", blocking=False)
    assert result is None


def test_therapy_is_independent_of_retrieval_keys() -> None:
    """
    Therapy engine must ignore retrieval/advisory context keys and remain driven
    only by perception-derived signals it consumes.
    """
    from ml.therapy import TherapyEngine, TherapyEngineConfig

    config = TherapyEngineConfig(stress_trigger_threshold=0.0, high_stress_threshold=0.35)
    engine = TherapyEngine(config=config)

    base_perception = {
        "detections": [{"class_name": "person"}] * 5,
        "uncertainty": 0.2,
        "navigation_difficulty": 0.9,
        "urgency": 2.0,
    }

    actions_without_retrieval = engine.update(base_perception)

    # Inject retrieval-like noise. Therapy should not crash and should keep
    # decisions consistent when the consumed signals are unchanged.
    engine2 = TherapyEngine(config=config)
    perception_with_retrieval = {**base_perception, "retrieval_results": {"any": "thing"}}
    actions_with_retrieval = engine2.update(perception_with_retrieval)

    assert len(actions_without_retrieval) == len(actions_with_retrieval)
    if actions_without_retrieval:
        assert (
            actions_without_retrieval[0].intervention_type
            == actions_with_retrieval[0].intervention_type
        )


def test_forward_triggers_retrieval_with_blocking_false() -> None:
    """
    Forward-path contract:
    - when retrieval is enabled, `MaxSightCNN.forward()` must request retrieval with blocking=False
    - forward must remain tensor-only (no retrieval artefacts in outputs)
    """
    from ml.models.maxsight_cnn import CapabilityTier, TierConfig, create_model

    tier_config = TierConfig.for_tier(CapabilityTier.T5_TEMPORAL)
    tier_config.use_retrieval = True
    # Disable cross-task scene graph so Stage B isn't skipped via scene graph invalidation.
    tier_config.use_cross_task_attention = False

    model = create_model(use_audio=False, tier_config=tier_config)
    model.eval()

    called = {"count": 0, "blocking": None}

    class FakeRetrievalSystem:
        def retrieve(self, query_embeddings, request_id=None, blocking=False):
            called["count"] += 1
            called["blocking"] = blocking
            return None

    model.retrieval_system = FakeRetrievalSystem()

    dummy_image = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        outputs = model(dummy_image)

    assert called["count"] >= 1
    assert called["blocking"] is False

    # Retrieval/advisory must not pollute the runtime tensor-only output surface.
    assert "retrieval_results" not in outputs
    assert "distances" not in outputs
    assert "indices" not in outputs
    # Scene description head stays disabled in forward to preserve trace/export safety.
    assert "scene_description" not in outputs
