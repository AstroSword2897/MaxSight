"""Tests for production vs simulator runtime selection."""

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.runtime.mode import (  # noqa: E402
    RuntimeMode,
    get_runtime_mode,
    is_production_runtime,
)


def test_default_is_simulator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MAXSIGHT_RUNTIME", raising=False)
    assert get_runtime_mode() == RuntimeMode.SIMULATOR
    assert is_production_runtime() is False


@pytest.mark.parametrize("value", ["production", "prod", "PRODUCTION"])
def test_production_aliases(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("MAXSIGHT_RUNTIME", value)
    assert get_runtime_mode() == RuntimeMode.PRODUCTION
    assert is_production_runtime() is True


def test_simulator_config_validation() -> None:
    from tools.simulation.config import SimulatorConfig

    with pytest.raises(ValueError):
        SimulatorConfig(port=0)
    with pytest.raises(ValueError):
        SimulatorConfig(confidence_threshold=1.5)
