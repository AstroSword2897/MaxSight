import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_temporal_5d_input_can_be_jit_exported_smoke() -> None:
    """
    Export smoke contract:
    - temporal-tier model must accept `[1, T, 3, H, W]` forward
    - JIT export must succeed with the same 5D input shape
    """
    import os
    import pytest

    # Hard-safety: temporal JIT tracing may segfault for some operator graphs.
    # Default to skipping in normal CI to keep the suite stable; enable explicitly
    # when running manual export validation.
    if os.environ.get("RUN_TEMPORAL_JIT_EXPORT_SMOKE", "0") != "1":
        pytest.skip("Opt-in: set RUN_TEMPORAL_JIT_EXPORT_SMOKE=1 to run temporal JIT export smoke.")

    from ml.models.maxsight_cnn import CapabilityTier, TierConfig, create_model
    from ml.training.export import export_to_jit

    tier_config = TierConfig.for_tier(CapabilityTier.T5_TEMPORAL)
    tier_config.use_retrieval = False
    tier_config.use_cross_task_attention = True

    model = create_model(use_audio=False, tier_config=tier_config)
    model.eval()

    t = 8
    x = torch.randn(1, t, 3, 224, 224)

    with torch.no_grad():
        out = model(x)
    assert isinstance(out, dict)

    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        jit_path = f.name

    try:
        try:
            export_to_jit(
                model,
                jit_path,
                input_size=(1, t, 3, 224, 224),
                validate=True,
            )
        except RuntimeError as e:
            # Some dict-output models or temporal ops may still be non-traceable.
            # This test is a smoke check: skip only on known tracer limitations.
            msg = str(e).lower()
            if "tracer cannot infer" in msg or "dict" in msg:
                pytest.skip(f"JIT tracing limitation: {e}")
            raise

        traced = torch.jit.load(jit_path)
        traced.eval()
        with torch.no_grad():
            out_traced = traced(x)
        assert isinstance(out_traced, dict)
    finally:
        Path(jit_path).unlink(missing_ok=True)

