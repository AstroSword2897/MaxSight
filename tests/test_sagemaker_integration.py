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
        if "health_summary" in mdef["Regex"]:
            continue
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


def test_sm_config_parses_vpc_and_kms_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from ml.infra.sagemaker_utils import SMConfig

    monkeypatch.setenv("SM_SUBNET_IDS", "subnet-a, subnet-b")
    monkeypatch.setenv("SM_SECURITY_GROUP_IDS", "sg-1,sg-2")
    monkeypatch.setenv("SM_VOLUME_KMS_KEY_ID", "arn:aws:kms:us-east-1:123:key/uuid")
    cfg = SMConfig.from_env()
    assert cfg.subnets == ("subnet-a", "subnet-b")
    assert cfg.security_group_ids == ("sg-1", "sg-2")
    assert cfg.volume_kms_key_id == "arn:aws:kms:us-east-1:123:key/uuid"


def test_health_summary_metrics_match_log_line() -> None:
    line = (
        "health_summary epoch=2 processed_batches=20 skipped_batches=2 skip_ratio=9.09% "
        "train_loss=0.8123 val_loss=0.7011 val_map=0.4321 new_best=True lr=1.000000e-04"
    )
    health_defs = [m for m in TRAINING_METRIC_DEFINITIONS if m["Name"].startswith("train:")]
    assert {"train:processed_batches", "train:skipped_batches", "train:skip_ratio_pct"}.issubset(
        {m["Name"] for m in health_defs}
    )
    for mdef in health_defs:
        if mdef["Name"] == "train:loss":
            continue
        if "health_summary" not in mdef["Regex"]:
            continue
        assert re.search(mdef["Regex"], line), f"No match for {mdef['Name']}"


def test_model_registry_reads_package_group_from_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from ml.infra.model_registry import ModelRegistry

    monkeypatch.setenv("MAXSIGHT_MODEL_PACKAGE_GROUP", "maxsight-test-group")
    reg = ModelRegistry(registry_path=tmp_path / "reg.json")
    assert reg._sm_group == "maxsight-test-group"
    monkeypatch.delenv("MAXSIGHT_MODEL_PACKAGE_GROUP", raising=False)
