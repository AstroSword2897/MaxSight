"""Behavioural regression tests for every schema in schemas/*.schema.json.

Layers: (1) a known-good document per schema, (2) generated mutations (missing required field,
extra field, wrong version), (3) targeted enum/range/format rejections, (4) house-style rules,
(5) spec gaps recorded as strict xfails so they fail loudly once the schema is fixed.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

jsonschema = pytest.importorskip("jsonschema")
pytest.importorskip("referencing")
ValidationError = jsonschema.ValidationError

SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schemas"
NAMES = sorted(p.name.removesuffix(".schema.json") for p in SCHEMA_DIR.glob("*.schema.json"))
RAW = {n: json.loads((SCHEMA_DIR / f"{n}.schema.json").read_text()) for n in NAMES}
REQUIRED_PAIRS = [(n, f) for n in NAMES for f in RAW[n].get("required", [])]
VERSIONED = [n for n in NAMES if "schema_version" in RAW[n].get("properties", {})]
GAP_NO_VERSION = {"stage-b-task"}  # known gap: request carries no schema_version


def _mutated(payloads, name, path, value):
    doc = copy.deepcopy(payloads[name])
    target = doc
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return doc


@pytest.mark.parametrize("name", NAMES)
def test_known_good_payload_validates(validator, payloads, name):
    validator(name).validate(payloads[name])


@pytest.mark.parametrize(("name", "field"), REQUIRED_PAIRS)
def test_missing_required_field_is_rejected(validator, payloads, name, field):
    doc = copy.deepcopy(payloads[name])
    del doc[field]
    with pytest.raises(ValidationError):
        validator(name).validate(doc)


@pytest.mark.parametrize("name", NAMES)
def test_unknown_top_level_field_is_rejected(validator, payloads, name):
    doc = payloads[name] | {"condition_mode": "glaucoma"}  # perception is condition-independent
    with pytest.raises(ValidationError):
        validator(name).validate(doc)


@pytest.mark.parametrize("name", VERSIONED)
def test_wrong_schema_version_is_rejected(validator, payloads, name):
    with pytest.raises(ValidationError):
        validator(name).validate(payloads[name] | {"schema_version": "maxsight.stage_a/1.0"})


REJECTIONS = [
    ("frame-metadata", ("orientation",), "upside_down"),
    ("frame-metadata", ("width",), 0),
    ("frame-metadata", ("frame_id",), 0),
    ("frame-quality", ("label",), "great"),
    ("frame-quality", ("luma_mean",), 256),
    ("frame-quality", ("clipped_frac",), 1.5),
    ("detection-batch", ("detections", 0, "score"), 1.2),
    ("detection-batch", ("detections", 0, "bbox_xyxy"), [1, 2, 3]),
    ("tracked-object", ("track_confidence",), -0.1),
    ("tracked-object", ("bbox_xyxy",), [1, 2, 3, 4, 5]),
    ("spatial-estimate", ("sector",), "behind"),
    ("spatial-estimate", ("azimuth_deg",), 91),
    ("spatial-estimate", ("distance_m",), 0),  # unknown distance must never become 0
    ("spatial-estimate", ("distance_m",), 9),
    ("spatial-estimate", ("distance_method",), "lidar"),  # camera-only product
    ("uncertainty-assessment", ("evidence",), "certain"),
    ("uncertainty-assessment", ("detector",), 2),
    ("hazard-assessment", ("urgency",), 4),
    ("hazard-assessment", ("category",), "ambient"),
    ("hazard-assessment", ("rule_ids",), []),
    ("hazard-event", ("state",), "CLOSED"),
    ("hazard-event", ("delivery",), "RESOLVED"),  # delivery is separate from hazard state
    ("hazard-event", ("event_id",), "not-a-uuid"),
    ("alert-decision", ("kind",), "marketing"),
    ("alert-decision", ("urgency",), -1),
    ("alert-decision", ("decision_id",), "nope"),
    ("environmental-state", ("system_status",), "OK"),
    ("environmental-state", ("platform",), "wearable"),
    ("environmental-state", ("quality", "label"), "great"),
    ("fvp", ("preferred_modality",), "braille"),
    ("fvp", ("speech", "rate_wpm"), 400),
    ("fvp", ("haptic", "intensity"), 0.1),
    ("stage-b-task", ("task_type",), "navigate"),
    ("stage-b-task", ("deadline_ms",), 4001),
    ("stage-b-task", ("request_id",), "not-a-uuid"),
]


@pytest.mark.parametrize(("name", "path", "value"), REJECTIONS,
                         ids=[f"{n}:{'.'.join(map(str, p))}={v!r}" for n, p, v in REJECTIONS])
def test_targeted_value_is_rejected(validator, payloads, name, path, value):
    with pytest.raises(ValidationError):
        validator(name).validate(_mutated(payloads, name, path, value))


@pytest.mark.parametrize(("name", "path", "value"), [
    ("spatial-estimate", ("distance_m",), None),
    ("spatial-estimate", ("distance_m",), 0.3),
    ("uncertainty-assessment", ("localization",), None),
    ("alert-decision", ("event_id",), None),
    ("frame-metadata", ("iso",), None),
    ("hazard-assessment", ("urgency",), 3),
])
def test_boundary_and_nullable_values_are_accepted(validator, payloads, name, path, value):
    validator(name).validate(_mutated(payloads, name, path, value))


def test_environmental_state_nested_item_errors_are_reported(validator, payloads):
    doc = _mutated(payloads, "environmental-state", ("hazards", 0, "urgency"), 9)
    errors = list(validator("environmental-state").iter_errors(doc))
    assert errors and "urgency" in "/".join(map(str, errors[0].absolute_path))


@pytest.mark.parametrize("field", ["image", "frame", "crop", "thumbnail", "embedding", "frames"])
def test_stage_b_never_accepts_visual_payload_fields(validator, payloads, field):
    doc = copy.deepcopy(payloads["stage-b-task"])
    doc["inputs"][field] = "AAAA"
    with pytest.raises(ValidationError):
        validator("stage-b-task").validate(doc)


# ---- house style -----------------------------------------------------------------------------

@pytest.mark.parametrize("name", NAMES)
def test_house_style(name):
    s = RAW[name]
    assert s["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert s["$id"] == f"https://maxsight.dev/schemas/{name}.schema.json"
    assert s["type"] == "object"
    assert s["additionalProperties"] is False
    assert s["title"] and s.get("description")


@pytest.mark.parametrize("name", [n for n in NAMES if n not in GAP_NO_VERSION])
def test_every_schema_pins_its_version(name):
    const = RAW[name]["properties"]["schema_version"]["const"]
    assert const in {"maxsight.stage_a/2.0", "maxsight.fvp/1.0"}
    assert "schema_version" in RAW[name]["required"]


def test_no_schema_declares_a_forbidden_sensor_or_condition_field():
    text = json.dumps(RAW).lower()
    for word in ("imu", "lidar", "gps", "accelerometer", "condition_mode", "diagnosis"):
        assert f'"{word}' not in text, word


# ---- known gaps: strict xfail, flip to failures when the schema is fixed ------------------------

def _xfail(reason):
    return pytest.mark.xfail(strict=True, reason=reason)


@_xfail("spec 03 D3.3: hazard-assessment carries uncertainty_ref; schema forbids it")
def test_gap_hazard_assessment_has_uncertainty_ref(validator, payloads):
    validator("hazard-assessment").validate(payloads["hazard-assessment"] | {"uncertainty_ref": 1})


@_xfail("spec: urgency 3 must interrupt; no cross-field rule yet")
def test_gap_critical_alert_must_interrupt(validator, payloads):
    doc = payloads["alert-decision"] | {"urgency": 3, "interrupt": False}
    with pytest.raises(ValidationError):
        validator("alert-decision").validate(doc)


@_xfail("spec: status announcements are not tied to a hazard event; event_id must be null")
def test_gap_status_announcement_has_no_event(validator, payloads):
    doc = payloads["alert-decision"] | {"kind": "status_announcement"}
    with pytest.raises(ValidationError):
        validator("alert-decision").validate(doc)


@_xfail("invariant 3: a poor-quality frame cannot imply CLEAR_VALID")
def test_gap_unusable_frame_cannot_be_clear_valid(validator, payloads):
    doc = copy.deepcopy(payloads["environmental-state"])
    doc["quality"] |= {"usable": False, "label": "blur"}
    doc["hazards"] = []
    with pytest.raises(ValidationError):
        validator("environmental-state").validate(doc)


@_xfail("timestamps should be format: date-time")
def test_gap_stage_b_timestamp_is_date_time(validator, payloads):
    doc = payloads["stage-b-task"] | {"client_ts": "yesterday-ish"}
    with pytest.raises(ValidationError):
        validator("stage-b-task").validate(doc)


@_xfail("fvp updated_at should be format: date-time")
def test_gap_fvp_timestamp_is_date_time(validator, payloads):
    with pytest.raises(ValidationError):
        validator("fvp").validate(payloads["fvp"] | {"updated_at": "whenever"})


@_xfail("DR-202 open: class_name should be a taxonomy enum once the hazard taxonomy is frozen")
def test_gap_class_name_is_taxonomy_enum(validator, payloads):
    doc = _mutated(payloads, "tracked-object", ("class_name",), "unicorn")
    with pytest.raises(ValidationError):
        validator("tracked-object").validate(doc)


@_xfail("stage-b-task has no schema_version (spec: maxsight.stageb/1.0)")
def test_gap_stage_b_request_pins_version():
    assert "schema_version" in RAW["stage-b-task"]["properties"]
