"""Shared helpers for schema contract tests: registry-aware validators and known-good payloads."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

# No module-level importorskip here: a skip at conftest import time aborts the whole pytest session
# in profiles without jsonschema. Test modules that need it skip themselves.
SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schemas"
STAGE_A_VERSION = "maxsight.stage_a/2.0"
SCHEMA_NAMES = (
    "frame-metadata", "frame-quality", "detection-batch", "tracked-object", "spatial-estimate",
    "uncertainty-assessment", "hazard-assessment", "hazard-event", "alert-decision",
    "environmental-state", "fvp", "stage-b-task",
)


def load_schemas() -> dict[str, dict]:
    return {n: json.loads((SCHEMA_DIR / f"{n}.schema.json").read_text()) for n in SCHEMA_NAMES}


def build_validator(name: str):
    from jsonschema import Draft202012Validator, FormatChecker
    from referencing import Registry, Resource

    schemas = load_schemas()
    registry = Registry().with_resources(
        (f"{n}.schema.json", Resource.from_contents(s)) for n, s in schemas.items()
    )
    return Draft202012Validator(schemas[name], registry=registry, format_checker=FormatChecker())


def _uuid() -> str:
    return str(uuid.uuid4())


def valid_payloads() -> dict[str, dict]:
    """One known-good document per schema. Each test copies before mutating."""
    v = STAGE_A_VERSION
    quality = {"schema_version": v, "frame_id": 1, "label": "ok", "usable": True,
               "luma_mean": 120.0, "clipped_frac": 0.0, "laplacian_var": 300.0, "flow_median_px": None}
    track = {"schema_version": v, "track_id": 1, "class_name": "person", "bbox_xyxy": [1, 2, 30, 40],
             "hits": 3, "misses": 0, "age_frames": 3, "track_confidence": 0.9, "last_det_id": None}
    spatial = {"schema_version": v, "track_id": 1, "azimuth_deg": -12.0, "sector": "center_left",
               "ground_x_m": None, "ground_z_m": None, "distance_m": None, "distance_sigma_m": None,
               "distance_method": "none", "approach_rate_per_s": None, "ttc_s": None,
               "ego_compensated": False, "unknown_reasons": ["no_calibration"]}
    uncertainty = {"schema_version": v, "track_id": 1, "detector": 0.8, "localization": None,
                   "tracking": 0.9, "spatial": None, "evidence": "partial", "calibrated": False}
    hazard = {"schema_version": v, "hazard_id": "h1", "track_id": 1, "category": "static",
              "path_relevance": "possible", "severity": "moderate", "urgency": 1, "qualified": True,
              "rule_ids": ["R6"]}
    event = {"schema_version": v, "event_id": _uuid(), "hazard_id": "h1", "state": "NEW", "urgency_max": 1,
             "first_seen_ns": 10, "last_update_ns": 10, "delivery": "PENDING"}
    decision = {"schema_version": v, "decision_id": _uuid(), "event_id": event["event_id"],
                "kind": "hazard_alert", "urgency": 1, "interrupt": False, "message_key": "hazard.static",
                "params": {}, "frame_id": 1, "expires_ns": 300_000_000}
    return {
        "frame-metadata": {"schema_version": v, "frame_id": 1, "t_mono_ns": 5, "width": 640, "height": 480,
                           "orientation": "portrait", "camera_id": "back-wide", "exposure_ms": None, "iso": None},
        "frame-quality": quality,
        "detection-batch": {"schema_version": v, "frame_id": 1, "model_version": "m-ssd-0.0.0",
                            "inference_ms": 11.0, "skipped_reason": None,
                            "detections": [{"det_id": "d1", "class_name": "person", "score": 0.7,
                                            "bbox_xyxy": [1, 2, 30, 40]}]},
        "tracked-object": track,
        "spatial-estimate": spatial,
        "uncertainty-assessment": uncertainty,
        "hazard-assessment": hazard,
        "hazard-event": event,
        "alert-decision": decision,
        "environmental-state": {
            "schema_version": v, "frame_id": 1, "t_mono_ns": 0, "model_version": "m-ssd-0.0.0",
            "system_status": "READY", "perception_state": "CLEAR_VALID", "processing_latency_ms": 12.5,
            "quality": quality, "tracks": [track], "spatial": [spatial], "uncertainty": [uncertainty],
            "hazards": [hazard], "events": [event], "decisions": [decision], "platform": "reference"},
        "fvp": {"schema_version": "maxsight.fvp/1.0", "profile_id": _uuid(), "preferred_modality": "both",
                "alert_verbosity": "standard", "speech": {"rate_wpm": 200, "voice_id": None, "volume": 1.0},
                "haptic": {"enabled": True, "intensity": 0.8},
                "assistance": {"proactive_context": False, "repeat_on_request": True, "describe_on_degraded": True},
                "consent": {"c1": True, "c2": False, "c3": False, "sync": False},
                "updated_at": "2026-11-02T00:00:00Z"},
        "stage-b-task": {"request_id": _uuid(), "session_id": _uuid(), "task_type": "scene",
                         "client_ts": "2026-11-02T00:00:00Z", "frame_ref": {"frame_id": 1, "t_mono_ns": 0},
                         "deadline_ms": 4000, "inputs": {"text": "what is ahead"}},
    }


@pytest.fixture(scope="session")
def schemas() -> dict[str, dict]:
    return load_schemas()


@pytest.fixture
def validator():
    return build_validator


@pytest.fixture
def payloads() -> dict[str, dict]:
    return valid_payloads()
