"""Offline dry-run tests for scripts/ops/ launcher scripts.

These tests invoke the launcher main() functions in dry-run or mock mode —
no boto3 calls, no SageMaker SDK, no network required. They exist to ensure
that Cursor-assisted edits to the thin launcher scripts cannot silently break
production deploy flows without a test catching it.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import scripts.ops.sagemaker_train as sagemaker_train
import scripts.ops.sagemaker_deploy as sagemaker_deploy


# ── helpers ────────────────────────────────────────────────────────────────────

def _mock_s3_client():
    """Return a MagicMock that satisfies S3Client usage in the train path."""
    m = MagicMock()
    m.upload_gold_index.return_value = "s3://b/maxsight/gold/training_index.json"
    m.upload_medallion_layer.return_value = {"files_uploaded": 0}
    return m


# ── test 1: train dry-run exits 0 and prints valid JSON ───────────────────────

_TIER_CONFIG = str(REPO / "ml" / "training" / "configs" / "t5_temporal.yaml")


def test_train_dry_run_exits_0_and_prints_json(capsys):
    """Dry-run must exit 0 and emit JSON containing job_name and hyperparameters."""
    with patch("scripts.ops.sagemaker_train.S3Client", return_value=_mock_s3_client()):
        sys.argv = [
            "sagemaker_train.py",
            "--bucket", "test-bucket",
            "--role", "arn:aws:iam::123456789012:role/SageMakerRole",
            "--config", _TIER_CONFIG,
            "--dry-run",
        ]
        rc = sagemaker_train.main()

    assert rc == 0
    out = capsys.readouterr().out
    json_start = out.index("{")
    parsed = json.loads(out[json_start:])
    assert "job_name" in parsed, "dry-run output must include job_name"
    assert "hyperparameters" in parsed, "dry-run output must include hyperparameters"
    # YAML config path must be passed through so SageMaker entry resolves it.
    assert parsed["hyperparameters"].get("config", "").endswith(".yaml")


# ── test 2: train dry-run warns to stderr when gold index is absent ────────────

def test_train_dry_run_warns_on_missing_gold(tmp_path, capsys):
    """When no gold index exists, a warning containing 'gold index' must appear on stderr."""
    with patch("scripts.ops.sagemaker_train.S3Client", return_value=_mock_s3_client()):
        sys.argv = [
            "sagemaker_train.py",
            "--bucket", "test-bucket",
            "--role", "arn:aws:iam::123456789012:role/SageMakerRole",
            "--config", _TIER_CONFIG,
            "--medallion-root", str(tmp_path),
            "--dry-run",
        ]
        rc = sagemaker_train.main()

    assert rc == 0
    err = capsys.readouterr().err.lower()
    assert "gold index" in err, (
        "stderr must mention 'gold index' when the gold index file is absent; "
        f"got: {err!r}"
    )


# ── test 3: deploy rejects an unregistered artifact ───────────────────────────

def test_deploy_rejects_unregistered_artifact(tmp_path):
    """cmd_deploy must return 1 when the artifact is not in the registry."""
    # Registry backed by an empty tmp directory — no entries.
    with patch("scripts.ops.sagemaker_deploy.ModelRegistry") as MockReg:
        instance = MockReg.return_value
        instance.list_models.return_value = []

        sys.argv = [
            "sagemaker_deploy.py",
            "--bucket", "b",
            "deploy",
            "--model-data", "s3://b/unregistered/model.tar.gz",
        ]
        rc = sagemaker_deploy.main()

    assert rc == 1, "deploy should return exit code 1 for an unregistered artifact"


# ── test 4: --skip-registry-check emits WARNING to stderr ─────────────────────

def test_deploy_skip_registry_check_emits_warning(capsys):
    """--skip-registry-check must always emit a WARNING to stderr, even on dry-run."""
    sys.argv = [
        "sagemaker_deploy.py",
        "--bucket", "b",
        "deploy",
        "--model-data", "s3://b/m.tar.gz",
        "--skip-registry-check",
        "--dry-run",
    ]
    sagemaker_deploy.main()
    err = capsys.readouterr().err
    assert "WARNING" in err, (
        "--skip-registry-check must print a WARNING to stderr; "
        f"got: {err!r}"
    )


def test_deploy_skip_registry_check_rejected_in_production(monkeypatch):
    """Production profile must not allow --skip-registry-check."""
    monkeypatch.setenv("MAXSIGHT_ENV", "production")
    sys.argv = [
        "sagemaker_deploy.py",
        "--bucket", "b",
        "deploy",
        "--model-data", "s3://b/m.tar.gz",
        "--skip-registry-check",
        "--dry-run",
    ]
    rc = sagemaker_deploy.main()
    assert rc == 1
    monkeypatch.delenv("MAXSIGHT_ENV", raising=False)


# ── test 5: sagemaker_entrypoint raises ImportError when imported ──────────────

def test_entrypoint_raises_importerror_on_import():
    """ml.pipeline.sagemaker_entrypoint must raise ImportError when imported, not run as __main__."""
    # Remove any cached module so the guard fires fresh.
    sys.modules.pop("ml.pipeline.sagemaker_entrypoint", None)

    with pytest.raises(ImportError, match="offline"):
        importlib.import_module("ml.pipeline.sagemaker_entrypoint")
