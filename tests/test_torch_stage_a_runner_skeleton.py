"""Skeleton / contract tests for TorchStageARunner (MAXS-101d + post-102c)."""

from __future__ import annotations

import pytest

from ml.runtime.stage_a import StageARunner
from ml.runtime.stage_a.torch_runner import TorchStageARunner


def test_torch_runner_is_stage_a_runner() -> None:
    runner = TorchStageARunner("/tmp/fake.pt")
    assert isinstance(runner, StageARunner)


def test_torch_runner_rejects_network_kwarg() -> None:
    with pytest.raises(TypeError):
        TorchStageARunner("/tmp/fake.pt", network_client=object())  # type: ignore[call-arg]


def test_constructing_runner_does_not_import_maxsight_cnn() -> None:
    import sys

    # Drop cached model module if present so we can assert lazy import.
    sys.modules.pop("ml.models.maxsight_cnn", None)
    TorchStageARunner("/tmp/fake.pt")
    assert "ml.models.maxsight_cnn" not in sys.modules
