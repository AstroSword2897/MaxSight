"""Contract tests for schemas/*.schema.json (Stage A, FVP, Stage B)."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

jsonschema = pytest.importorskip("jsonschema")
pytest.importorskip("referencing")

SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schemas"
V = "maxsight.stage_a/2.0"
EXPECTED = {
    "frame-metadata", "frame-quality", "detection-batch", "tracked-object", "spatial-estimate",
    "uncertainty-assessment", "hazard-assessment", "hazard-event", "alert-decision",
    "environmental-state", "fvp", "stage-b-task",
}


def _load() -> dict[str, dict]:
    return {
        p.name.removesuffix(".schema.json"): json.loads(p.read_text())
        for p in SCHEMA_DIR.glob("*.schema.json")
    }


def _validator(name: str):
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource

    schemas = _load()
    registry = Registry().with_resources(
        (f"{n}.schema.json", Resource.from_contents(s)) for n, s in schemas.items()
    )
    return Draft202012Validator(schemas[name], registry=registry)


def test_all_twelve_schemas_exist():
    assert set(_load()) == EXPECTED


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_schema_is_valid_draft_2020_12(name):
    from jsonschema import Draft202012Validator

    Draft202012Validator.check_schema(_load()[name])


def test_spatial_estimate_allows_unknown_distance_with_reason():
    doc = {"schema_version": V, "track_id": 3, "azimuth_deg": 4.2, "sector": "center", "distance_m": None,
           "distance_method": "none", "ego_compensated": False, "unknown_reasons": ["no_calibration"]}
    _validator("spatial-estimate").validate(doc)


def test_spatial_estimate_rejects_unknown_field():
    doc = {"schema_version": V, "track_id": 3, "azimuth_deg": 4.2, "sector": "center",
           "distance_method": "none", "ego_compensated": False, "unknown_reasons": [], "color": "red"}
    with pytest.raises(jsonschema.ValidationError):
        _validator("spatial-estimate").validate(doc)


def test_urgency_out_of_range_rejected():
    doc = {"schema_version": V, "hazard_id": "h1", "track_id": 1, "category": "static",
           "path_relevance": "none", "severity": "low", "urgency": 4, "qualified": False,
           "rule_ids": ["R6"]}
    with pytest.raises(jsonschema.ValidationError):
        _validator("hazard-assessment").validate(doc)


def _empty_state() -> dict:
    return {
        "schema_version": V, "frame_id": 1, "t_mono_ns": 0, "model_version": "m-ssd-0.0.0",
        "system_status": "READY", "perception_state": "CLEAR_VALID", "processing_latency_ms": 12.5,
        "quality": {"schema_version": V, "frame_id": 1, "label": "ok", "usable": True,
                    "luma_mean": 120.0, "clipped_frac": 0.0, "laplacian_var": 300.0},
        "tracks": [], "spatial": [], "uncertainty": [], "hazards": [], "events": [], "decisions": [],
        "platform": "reference",
    }


def test_environmental_state_resolves_refs():
    _validator("environmental-state").validate(_empty_state())


def test_environmental_state_rejects_bad_perception_state():
    doc = _empty_state() | {"perception_state": "ALL_CLEAR"}
    with pytest.raises(jsonschema.ValidationError):
        _validator("environmental-state").validate(doc)


def _task(**inputs) -> dict:
    return {"request_id": str(uuid.uuid4()), "session_id": str(uuid.uuid4()), "task_type": "scene",
            "client_ts": "2026-11-02T00:00:00Z", "frame_ref": {"frame_id": 1, "t_mono_ns": 0},
            "deadline_ms": 4000, "inputs": inputs}


def test_stage_b_accepts_text_inputs():
    _validator("stage-b-task").validate(_task(text="read this", scene={"objects": []}))


@pytest.mark.parametrize("field", ["image", "frame", "crop", "thumbnail", "embedding"])
def test_stage_b_rejects_visual_inputs_under_egress_policy_v1(field):
    with pytest.raises(jsonschema.ValidationError):
        _validator("stage-b-task").validate(_task(**{field: "AAAA"}))


def test_stage_b_deadline_capped_at_four_seconds():
    with pytest.raises(jsonschema.ValidationError):
        _validator("stage-b-task").validate(_task() | {"deadline_ms": 5000})


def test_fvp_defaults_validate():
    doc = {"schema_version": "maxsight.fvp/1.0", "profile_id": str(uuid.uuid4()),
           "preferred_modality": "both", "alert_verbosity": "standard",
           "speech": {"rate_wpm": 200, "voice_id": None, "volume": 1.0},
           "haptic": {"enabled": True, "intensity": 0.8},
           "assistance": {"proactive_context": False, "repeat_on_request": True, "describe_on_degraded": True},
           "consent": {"c1": True, "c2": False, "c3": False, "sync": False},
           "updated_at": "2026-11-02T00:00:00Z"}
    _validator("fvp").validate(doc)
