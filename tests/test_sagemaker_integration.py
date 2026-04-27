"""Offline tests for SageMaker helpers (no AWS calls)."""

from __future__ import annotations

import re
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.infra.sagemaker_utils import TRAINING_METRIC_DEFINITIONS  # noqa: E402


def test_training_metric_regexes_match_log_line() -> None:
    line = (
        "Train Loss: 1.2345, Val Loss: 2.3456, Val mAP: 0.4567, "
        "Val mAP@0.5: 0.5000, Val mAP@0.75: 0.4000, Precision: 0.55, Recall: 0.66, F1: 0.61"
    )
    for mdef in TRAINING_METRIC_DEFINITIONS:
        rx = re.compile(mdef["Regex"])
        assert rx.search(line), f"No match for {mdef['Name']}: {mdef['Regex']}"


def test_pipeline_config_prefers_env_over_processing_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from ml.pipeline.sagemaker_config import SageMakerPipelineConfig

    d = tmp_path / "train_in"
    d.mkdir(parents=True)
    monkeypatch.setenv("SM_CHANNEL_TRAIN", str(d))
    monkeypatch.delenv("SM_OUTPUT_DATA_DIR", raising=False)
    monkeypatch.delenv("SM_MODEL_DIR", raising=False)
    cfg = SageMakerPipelineConfig.from_env()
    assert cfg.input_dir == d.resolve()


def test_default_source_dir_points_at_repo() -> None:
    from ml.infra.sagemaker_utils import _default_source_dir

    root = Path(_default_source_dir())
    assert (root / "ml" / "training" / "sagemaker_entry.py").is_file()


def test_metric_definition_names_are_unique() -> None:
    names = [m["Name"] for m in TRAINING_METRIC_DEFINITIONS]
    assert len(names) == len(set(names))
