"""Runnable checks for video manifest, temporal targets, collate, and temporal losses (simulator dev API)."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

# Tests avoid pytest so the simulator can import and run without dev deps layout issues.


def _ok(name: str, detail: str = "") -> dict[str, Any]:
    return {"name": name, "ok": True, "detail": detail, "ms": 0.0}


def _fail(name: str, err: str) -> dict[str, Any]:
    return {"name": name, "ok": False, "detail": err, "ms": 0.0}


def _timed(name: str, fn: Callable[[], None]) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        fn()
        ms = (time.perf_counter() - t0) * 1000.0
        r = _ok(name)
        r["ms"] = round(ms, 2)
        return r
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000.0
        r = _fail(name, f"{type(e).__name__}: {e}")
        r["ms"] = round(ms, 2)
        return r


def run_sprint_self_tests() -> dict[str, Any]:
    """Run all sprint-related self-tests; return JSON-serializable report."""

    results: list[dict[str, Any]] = []

    def t_manifest_validate() -> None:
        from ml.data.video_manifest import validate_manifest_v1

        data = {
            "schema_version": "1.0",
            "clips": [
                {
                    "clip_id": "x",
                    "video_id": "v",
                    "start_frame": 0,
                    "end_frame": 8,
                    "temporal_window": 8,
                    "temporal_stride": 1,
                    "temporal_overlap": 0,
                    "frame_paths": [f"a{i}.jpg" for i in range(8)],
                    "frames_segments": [[] for _ in range(8)],
                }
            ],
        }
        assert validate_manifest_v1(data) == []
        assert validate_manifest_v1(data, require_fixed_t8=True) == []

    results.append(_timed("video_manifest.validate_manifest_v1", t_manifest_validate))

    def t_temporal_targets() -> None:
        from ml.data.temporal_clip_targets import derive_temporal_clip_targets

        b = {"bbox": [10.0, 10.0, 20.0, 20.0], "track_proxy_id": 1}
        tt = derive_temporal_clip_targets([[b], [b]])
        assert tt.temporal_consistency > 0.9

    results.append(_timed("temporal_clip_targets.derive", t_temporal_targets))

    def t_collate_and_loss() -> None:
        from ml.data.data_pipeline import collate_fn
        from ml.training.losses import MultiHeadLoss, ScalarMSELoss

        clip = {
            "frames": torch.randn(8, 3, 224, 224),
            "labels": torch.zeros(10, dtype=torch.long),
            "boxes": torch.zeros(10, 4),
            "distance": torch.zeros(10, dtype=torch.long),
            "num_objects": torch.tensor(0, dtype=torch.long),
            "urgency": torch.tensor(0, dtype=torch.long),
            "temporal_consistency": torch.tensor([0.8], dtype=torch.float32),
            "flicker": torch.tensor([0.1], dtype=torch.float32),
            "clip_id": "c0",
        }
        batch = collate_fn([clip, clip])
        assert batch["images"].shape == (2, 8, 3, 224, 224)
        assert "temporal_consistency" in batch

        loss_fn = MultiHeadLoss(
            {
                "temporal_consistency": ScalarMSELoss(),
                "flicker": ScalarMSELoss(),
            }
        )
        preds = {
            "temporal_consistency": torch.tensor([[0.7], [0.7]], dtype=torch.float32),
            "flicker": torch.tensor([[0.2], [0.2]], dtype=torch.float32),
        }
        targs = {
            "temporal_consistency": batch["temporal_consistency"],
            "flicker": batch["flicker"],
        }
        out = loss_fn(preds, targs)
        assert out["total_loss"].numel() == 1
        assert float(out["total_loss"].item()) >= 0.0

    results.append(_timed("collate_fn + MultiHeadLoss temporal", t_collate_and_loss))

    def t_validators_frames() -> None:
        from tools.simulation.validators import validate_frames_data  # type: ignore

        # API passes a list of base64 strings (optional data URL prefix stripped inside validator).
        tiny_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
        validate_frames_data([tiny_png], max_frames=16, max_payload_mb=10.0)

    results.append(_timed("validators.validate_frames_data", t_validators_frames))

    def t_sample_script_import() -> None:
        import importlib.util

        p = Path(__file__).resolve().parents[2] / "scripts" / "ops" / "sample_video_clips.py"
        spec = importlib.util.spec_from_file_location("sample_video_clips", p)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "build_manifest")

    results.append(_timed("scripts.sample_video_clips import", t_sample_script_import))

    passed = sum(1 for r in results if r["ok"])
    return {
        "ok": passed == len(results),
        "passed": passed,
        "total": len(results),
        "results": results,
    }


def run_manifest_json_check(manifest_json: str) -> dict[str, Any]:
    """Validate user-pasted JSON manifest (optional body for POST)."""

    from ml.data.video_manifest import validate_manifest_v1

    try:
        data = json.loads(manifest_json)
    except json.JSONDecodeError as e:
        return {"ok": False, "errors": [str(e)]}
    errs = validate_manifest_v1(data)
    return {"ok": len(errs) == 0, "errors": errs}
