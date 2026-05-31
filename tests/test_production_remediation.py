"""Integration tests for production remediation priorities."""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ui.haptic_backends import LogHapticBackend, NoopHapticBackend, resolve_haptic_backend
from app.ui.haptic_feedback import HapticFeedback, HapticPattern
from ml.therapy.therapy_engine import TherapyEngine
from ml.training.observability import (
    DEFAULT_MAX_SKIPPED_BATCH_RATIO,
    parse_health_summary_line,
    validate_skipped_batch_ratio,
)
from ml.training.run_config import ResolvedTrainingConfig
from ml.training.sagemaker_entry import apply_sagemaker_channel_paths
from ml.training.train_loop import ProductionTrainLoop


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.fc(x.mean(dim=(2, 3)))
        return {
            "classifications": logits.unsqueeze(1).expand(-1, 2, -1),
            "boxes": torch.zeros(x.shape[0], 2, 4),
            "objectness": torch.zeros(x.shape[0], 2),
        }


class _DummyLoss(nn.Module):
    def forward(
        self, outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return outputs["classifications"].sum() * 0.0 + 1.0


def _make_train_loop(tmp_path: Path) -> ProductionTrainLoop:
    images = torch.randn(4, 3, 8, 8)
    dataset = TensorDataset(images)
    loader = DataLoader(dataset, batch_size=2)

    def collate(batch: list[Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        imgs = torch.stack([item[0] for item in batch])
        b = imgs.shape[0]
        return imgs, {
            "boxes": torch.zeros(b, 2, 4),
            "labels": torch.zeros(b, 2, dtype=torch.long),
            "num_objects": torch.zeros(b, dtype=torch.long),
        }

    train_loader = DataLoader(dataset, batch_size=2, collate_fn=collate)
    return ProductionTrainLoop(
        model=_TinyModel(),
        train_loader=train_loader,
        val_loader=None,
        loss_fn=_DummyLoss(),
        device="cpu",
        num_epochs=1,
        checkpoint_dir=str(tmp_path / "ckpt"),
        use_mixed_precision=False,
        warmup_epochs=0,
    )


def test_load_checkpoint_missing_returns_false(tmp_path: Path) -> None:
    loop = _make_train_loop(tmp_path)
    assert loop._load_checkpoint(str(tmp_path / "missing.pt")) is False
    assert loop.checkpoint_resume_status == "missing"


def test_load_checkpoint_corrupt_raises(tmp_path: Path) -> None:
    loop = _make_train_loop(tmp_path)
    corrupt = tmp_path / "corrupt.pt"
    buffer = io.BytesIO()
    torch.save({"epoch": 0, "model_state_dict": loop.model.state_dict()}, buffer)
    corrupt.write_bytes(buffer.getvalue()[:32])
    with pytest.raises(Exception):
        loop._load_checkpoint(str(corrupt))
    assert loop.checkpoint_resume_status in {"corrupt", "failed"}


def test_resume_from_corrupt_checkpoint_raises_in_init(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.pt"
    buffer = io.BytesIO()
    torch.save({"epoch": 0, "model_state_dict": _TinyModel().state_dict()}, buffer)
    corrupt.write_bytes(buffer.getvalue()[:32])

    images = torch.randn(2, 3, 8, 8)
    dataset = TensorDataset(images)

    def collate(batch: list[Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        imgs = torch.stack([item[0] for item in batch])
        b = imgs.shape[0]
        return imgs, {
            "boxes": torch.zeros(b, 2, 4),
            "labels": torch.zeros(b, 2, dtype=torch.long),
            "num_objects": torch.zeros(b, dtype=torch.long),
        }

    train_loader = DataLoader(dataset, batch_size=2, collate_fn=collate)
    with pytest.raises(Exception):
        ProductionTrainLoop(
            model=_TinyModel(),
            train_loader=train_loader,
            val_loader=None,
            loss_fn=_DummyLoss(),
            device="cpu",
            num_epochs=1,
            checkpoint_dir=str(tmp_path / "ckpt"),
            resume_from=str(corrupt),
            use_mixed_precision=False,
            warmup_epochs=0,
        )


def test_skipped_batch_ratio_contract() -> None:
    validate_skipped_batch_ratio(1, 10, max_ratio=DEFAULT_MAX_SKIPPED_BATCH_RATIO)
    with pytest.raises(RuntimeError):
        validate_skipped_batch_ratio(2, 10, max_ratio=0.1)


def test_health_summary_parser_roundtrip() -> None:
    line = (
        "health_summary epoch=3 processed_batches=12 skipped_batches=2 skip_ratio=14.29% "
        "train_loss=0.7500 val_loss=0.6200 val_map=0.4100 new_best=True lr=1.000000e-04"
    )
    parsed = parse_health_summary_line(line)
    assert parsed.epoch == 3
    assert parsed.processed_batches == 12
    assert parsed.skipped_batches == 2
    assert parsed.new_best is True
    assert parsed.val_map == pytest.approx(0.41)


def _high_stress_perception(**overrides: Any) -> dict[str, Any]:
    base = {
        "detections": [{"class_name": "person"} for _ in range(12)],
        "uncertainty": 0.4,
        "navigation_difficulty": 1.0,
        "audio_environment": 1.0,
        "urgency": 3.0,
        "temporal_consistency": 0.4,
    }
    base.update(overrides)
    return base


def test_therapy_engine_closed_loop_on_user_response() -> None:
    engine = TherapyEngine()
    perception_before = _high_stress_perception()
    actions = engine.update(perception_before)
    assert actions, "Expected at least one therapeutic action with high-stress perception."

    memory_before = dict(engine.memory.long_term.successful_interventions)
    perception_after = {
        **perception_before,
        "uncertainty": 0.2,
        "navigation_difficulty": 0.3,
        "temporal_consistency": 0.8,
    }
    engine.on_user_response(perception_after)
    memory_after = engine.memory.long_term.successful_interventions
    assert memory_after != memory_before or engine.adaptation is not None


def test_simulator_therapy_response_flow_unit() -> None:
    """Verify awaiting-response flag drives on_user_response before next update."""
    engine = TherapyEngine()
    awaiting = False
    calls: list[str] = []

    def update(perception: dict[str, Any]) -> list[Any]:
        calls.append("update")
        return engine.update(perception)

    def on_user_response(perception: dict[str, Any]) -> None:
        calls.append("on_user_response")
        engine.on_user_response(perception)

    p1 = _high_stress_perception()
    p2 = _high_stress_perception(
        uncertainty=0.2, navigation_difficulty=0.2, audio_environment=0.1, temporal_consistency=0.9
    )

    actions = update(p1)
    awaiting = bool(actions)
    assert awaiting

    if awaiting:
        on_user_response(p2)
        awaiting = False
    update(p2)

    assert calls == ["update", "on_user_response", "update"]


def test_apply_sagemaker_channel_paths(tmp_path: Path) -> None:
    train = tmp_path / "train"
    val = tmp_path / "val"
    train.mkdir()
    val.mkdir()
    (train / "maxsight_train.json").write_text("{}", encoding="utf-8")
    (val / "maxsight_val.json").write_text("{}", encoding="utf-8")

    resolved = ResolvedTrainingConfig.from_sources(
        PROJECT_ROOT / "ml/training/configs/t0_baseline.yaml",
        cli_overrides={"run_id": "test-sm-channels", "experiment": "test"},
    )
    remapped = apply_sagemaker_channel_paths(resolved, train, val)
    assert remapped.data.train_annotation_file == str(train / "maxsight_train.json")
    assert remapped.data.val_annotation_file == str(val / "maxsight_val.json")
    assert remapped.data.image_dir == str(train)


def test_sagemaker_run_remaps_channels(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from ml.training import sagemaker_entry

    train = tmp_path / "train"
    val = tmp_path / "val"
    train.mkdir()
    val.mkdir()
    (train / "maxsight_train.json").write_text("{}", encoding="utf-8")
    (val / "maxsight_val.json").write_text("{}", encoding="utf-8")

    captured: dict[str, Any] = {}

    def fake_run_training(resolved, **kwargs):  # type: ignore[no-untyped-def]
        captured["train"] = resolved.data.train_annotation_file
        captured["val"] = resolved.data.val_annotation_file
        captured["image_dir"] = resolved.data.image_dir
        return {"best_model_path": str(tmp_path / "best_model.pt")}

    class _FakeTracker:
        def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            pass

        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *args):  # type: ignore[no-untyped-def]
            return False

        def log_params(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            return None

        def log_dataset_provenance(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            return None

        def log_artefact(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            return None

        def best_metric(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            return 0.0

    config_path = PROJECT_ROOT / "ml/training/configs/t0_baseline.yaml"
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True)
    (tmp_path / "output").mkdir(parents=True)
    monkeypatch.setattr(sys, "argv", ["sagemaker_entry", "--config", str(config_path)])
    monkeypatch.setattr(
        sagemaker_entry, "channel_dir", lambda name, fallback="": train if name == "train" else val
    )
    monkeypatch.setattr(sagemaker_entry, "run_training", fake_run_training)
    monkeypatch.setattr(sagemaker_entry, "RunTracker", _FakeTracker)
    monkeypatch.setattr(sagemaker_entry, "model_dir", lambda: model_dir)
    monkeypatch.setattr(sagemaker_entry, "output_dir", lambda: tmp_path / "output")

    sagemaker_entry.run(sagemaker_entry.parse_args())

    assert captured["train"] == str(train / "maxsight_train.json")
    assert captured["val"] == str(val / "maxsight_val.json")
    assert captured["image_dir"] == str(train)


def test_haptic_backend_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    assert isinstance(resolve_haptic_backend("none"), NoopHapticBackend)
    assert isinstance(resolve_haptic_backend("log"), LogHapticBackend)
    monkeypatch.setattr("app.ui.haptic_backends.platform.system", lambda: "Linux")
    assert isinstance(resolve_haptic_backend("auto", allow_log_fallback=True), LogHapticBackend)


def test_haptic_feedback_uses_injected_backend() -> None:
    backend = LogHapticBackend()
    haptic = HapticFeedback(enabled=True, backend=backend, allow_log_fallback=True)
    haptic.trigger(HapticPattern.MICRO_PULSE, intensity=0.4)
    haptic.stop()
    assert haptic.backend_name == "LogHapticBackend"
