"""Tests for the ML lifecycle: S3 client, experiment tracker, model registry.

All tests are offline — boto3/SageMaker calls are mocked so no real AWS
credentials or network access is required.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ml.infra.experiment_tracker import RunRecord, RunTracker, leaderboard, load_all_runs
from ml.infra.model_registry import ModelEntry, ModelRegistry
from ml.infra.s3_client import S3Client, is_s3_uri, parse_s3_uri, s3_uri_to_local
from ml.infra.s3_validation import MAX_OBJECT_KEY_BYTES, S3ValidationError, validate_object_key


# ── S3 helpers (pure Python — no boto3 needed) ─────────────────────────────────

def test_parse_s3_uri() -> None:
    b, k = parse_s3_uri("s3://my-bucket/some/prefix/file.json")
    assert b == "my-bucket"
    assert k == "some/prefix/file.json"


def test_parse_s3_uri_no_key() -> None:
    b, k = parse_s3_uri("s3://my-bucket/")
    assert b == "my-bucket"
    assert k == ""


def test_parse_s3_uri_invalid_raises() -> None:
    with pytest.raises(S3ValidationError):
        parse_s3_uri("https://example.com/bucket/key")
    with pytest.raises(S3ValidationError):
        parse_s3_uri("s3://")
    with pytest.raises(S3ValidationError):
        parse_s3_uri("s3://ab/x")  # bucket name too short


def test_is_s3_uri() -> None:
    assert is_s3_uri("s3://bucket/key")
    assert not is_s3_uri("/local/path")
    assert not is_s3_uri("https://example.com")


def test_s3_uri_to_local() -> None:
    base = Path("/data")
    local = s3_uri_to_local("s3://my-bucket/a/b/c.json", base)
    assert str(local) == "/data/a/b/c.json"


# ── S3Client (mocked boto3) ────────────────────────────────────────────────────

def _make_s3_client(mock_s3) -> S3Client:
    from ml.infra.s3_validation import MAX_SINGLE_PUT_BYTES, validate_bucket_name, validate_prefix

    session = MagicMock()
    session.client.return_value = mock_s3
    session.region_name = "us-east-1"
    client = S3Client.__new__(S3Client)
    client.bucket = validate_bucket_name("test-bucket")
    client.prefix = validate_prefix("maxsight")
    client._session = session
    client._s3 = mock_s3
    client._region = "us-east-1"
    client._max_attempts = 5
    client._max_upload_bytes = MAX_SINGLE_PUT_BYTES
    return client


def test_s3_key_format() -> None:
    mock_s3 = MagicMock()
    client = _make_s3_client(mock_s3)
    assert client._s3_key("medallion", "gold") == "maxsight/medallion/gold"


def test_s3_uri_format() -> None:
    mock_s3 = MagicMock()
    client = _make_s3_client(mock_s3)
    assert client._s3_uri("checkpoints", "run1") == "s3://test-bucket/maxsight/checkpoints/run1"


def test_upload_file_calls_boto(tmp_path: Path) -> None:
    mock_s3 = MagicMock()
    client = _make_s3_client(mock_s3)
    f = tmp_path / "model.pt"
    f.write_bytes(b"fake weights")
    uri = client.upload_file(f, "maxsight/checkpoints/run1/best.pt")
    mock_s3.upload_file.assert_called_once()
    assert uri.startswith("s3://test-bucket/")


def test_sync_upload_calls_upload_for_each_file(tmp_path: Path) -> None:
    mock_s3 = MagicMock()
    client = _make_s3_client(mock_s3)
    d = tmp_path / "data"
    d.mkdir()
    for i in range(3):
        (d / f"file_{i}.json").write_text("{}")
    result = client.sync_upload(d, "maxsight/gold", overwrite=True)
    assert mock_s3.upload_file.call_count == 3
    assert len(result.uris) == 3
    assert not result.failed


def test_list_keys_paginates(tmp_path: Path) -> None:
    mock_s3 = MagicMock()
    paginator = MagicMock()
    paginator.paginate.return_value = [
        {"Contents": [{"Key": "a/b.pt"}, {"Key": "a/c.pt"}]},
        {"Contents": [{"Key": "a/d.pt"}]},
    ]
    mock_s3.get_paginator.return_value = paginator
    client = _make_s3_client(mock_s3)
    keys = client.list_keys("a/")
    assert len(keys) == 3


def test_medallion_s3_prefix() -> None:
    mock_s3 = MagicMock()
    client = _make_s3_client(mock_s3)
    assert client.medallion_s3_prefix("silver") == "maxsight/medallion/silver"


def test_validate_object_key_too_long() -> None:
    key = "a" * (MAX_OBJECT_KEY_BYTES + 10)
    with pytest.raises(S3ValidationError):
        validate_object_key(key)


def test_s3_client_invalid_bucket_raises() -> None:
    mock_s3 = MagicMock()
    with pytest.raises(S3ValidationError):
        S3Client(bucket="X", prefix="p", session=MagicMock())


def test_sync_upload_continue_on_error_collects_failures(tmp_path: Path) -> None:
    mock_s3 = MagicMock()

    def upload_side_effect(local, bucket, key):
        if "bad" in key:
            raise OSError("simulated disk read error")

    mock_s3.upload_file.side_effect = upload_side_effect
    client = _make_s3_client(mock_s3)
    d = tmp_path / "data"
    d.mkdir()
    (d / "good.json").write_text("{}")
    bad_dir = d / "sub"
    bad_dir.mkdir()
    (bad_dir / "bad.json").write_text("{}")
    result = client.sync_upload(d, "prefix", overwrite=True, continue_on_error=True)
    assert len(result.failed) >= 1
    assert any("bad" in f.get("path", "") for f in result.failed)


# ── SMConfig ──────────────────────────────────────────────────────────────────

def test_smconfig_output_path() -> None:
    from ml.infra.sagemaker_utils import SMConfig

    cfg = SMConfig(bucket="my-bucket", prefix="test", region="us-east-1")
    assert cfg.output_path == "s3://my-bucket/test/output"


def test_smconfig_checkpoint_path() -> None:
    from ml.infra.sagemaker_utils import SMConfig

    cfg = SMConfig(bucket="my-bucket", prefix="test")
    assert "checkpoints" in cfg.checkpoint_s3_path


def test_smconfig_from_env(monkeypatch) -> None:
    from ml.infra.sagemaker_utils import SMConfig

    monkeypatch.setenv("MAXSIGHT_S3_BUCKET", "env-bucket")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-west-1")
    cfg = SMConfig.from_env()
    assert cfg.bucket == "env-bucket"
    assert cfg.region == "eu-west-1"


# ── RunTracker ────────────────────────────────────────────────────────────────

def test_run_tracker_creates_run_json(tmp_path: Path) -> None:
    with RunTracker(run_id="test_run_001", runs_dir=tmp_path) as run:
        run.log_params({"lr": 1e-4, "epochs": 10})
        run.log_metric("train_loss", 0.5, step=0)
        run.log_metric("train_loss", 0.3, step=1)
        run.log_metric("val_map", 0.45, step=1)

    run_json = tmp_path / "maxsight" / "test_run_001" / "run.json"
    assert run_json.exists()
    record = json.loads(run_json.read_text())
    assert record["status"] == "completed"
    assert record["params"]["lr"] == 1e-4
    assert len(record["metrics"]) == 3


def test_run_tracker_metrics_jsonl(tmp_path: Path) -> None:
    with RunTracker(run_id="test_metrics_stream", runs_dir=tmp_path) as run:
        for i in range(5):
            run.log_metric("loss", 1.0 - i * 0.1, step=i)

    metrics_file = tmp_path / "maxsight" / "test_metrics_stream" / "metrics.jsonl"
    assert metrics_file.exists()
    lines = [json.loads(l) for l in metrics_file.read_text().strip().split("\n")]
    assert len(lines) == 5


def test_run_tracker_marks_failed_on_exception(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        with RunTracker(run_id="test_fail", runs_dir=tmp_path):
            raise ValueError("training exploded")

    run_json = tmp_path / "maxsight" / "test_fail" / "run.json"
    record = json.loads(run_json.read_text())
    assert record["status"] == "failed"


def test_run_tracker_best_metric(tmp_path: Path) -> None:
    with RunTracker(run_id="test_best", runs_dir=tmp_path) as run:
        run.log_metric("val_map", 0.3, step=0)
        run.log_metric("val_map", 0.5, step=1)
        run.log_metric("val_map", 0.45, step=2)
        best = run.best_metric("val_map", mode="max")
    assert best == 0.5


def test_leaderboard(tmp_path: Path) -> None:
    for i in range(3):
        with RunTracker(run_id=f"lb_run_{i}", experiment="test", runs_dir=tmp_path) as run:
            run.log_metric("val_map", float(i) * 0.1, step=0)

    board = leaderboard(runs_dir=tmp_path, metric="val_map", mode="max", top_n=5)
    assert board[0]["run_id"] == "lb_run_2"  # highest val_map


# ── ModelRegistry ─────────────────────────────────────────────────────────────

def test_model_registry_register_and_list(tmp_path: Path) -> None:
    registry = ModelRegistry(registry_path=tmp_path / "registry.json")
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"fake weights")

    entry = registry.register_model(
        "run_001", ckpt, metrics={"val_map": 0.52}, tier="T5_TEMPORAL"
    )
    assert entry.stage == "candidate"
    assert entry.run_id == "run_001"
    assert len(registry.list_models()) == 1


def test_model_registry_promote_production(tmp_path: Path) -> None:
    registry = ModelRegistry(registry_path=tmp_path / "registry.json")
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"fake")

    registry.register_model("run_A", ckpt, metrics={"val_map": 0.4}, tier="T2_DETECTOR")
    registry.register_model("run_B", ckpt, metrics={"val_map": 0.55}, tier="T5_TEMPORAL")
    registry.promote_model("run_A", "production")
    registry.promote_model("run_B", "production")

    prod = registry.get_production_model()
    assert prod is not None
    assert prod.run_id == "run_B"
    archived = registry.get_stage_models("archived")
    assert any(e.run_id == "run_A" for e in archived)


def test_model_registry_persistence(tmp_path: Path) -> None:
    reg_path = tmp_path / "registry.json"
    r1 = ModelRegistry(registry_path=reg_path)
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"fake")
    r1.register_model("run_persisted", ckpt, tier="T3_MULTI_TASK")

    # Reload from disk.
    r2 = ModelRegistry(registry_path=reg_path)
    assert "run_persisted" in {e.run_id for e in r2.list_models()}


def test_model_registry_compare(tmp_path: Path) -> None:
    registry = ModelRegistry(registry_path=tmp_path / "registry.json")
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"fake")

    for i, score in enumerate([0.3, 0.5, 0.4]):
        registry.register_model(f"run_{i}", ckpt, metrics={"val_map": score})

    rows = registry.compare_models(metric="val_map")
    assert rows[0]["val_map"] == 0.5
