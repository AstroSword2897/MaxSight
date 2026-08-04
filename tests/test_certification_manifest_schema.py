"""Schema tests for certification manifest (MAXS-201a)."""

from __future__ import annotations

import json
from pathlib import Path

SCHEMA = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "contracts"
    / "schemas"
    / "model_certification_manifest.json"
)


def _validate(instance: dict) -> None:
    try:
        import jsonschema
    except ImportError:
        # Fallback: required keys only when jsonschema is unavailable.
        required = {"schema_version", "artifact_hash", "platform", "cells", "all_passed", "summary"}
        missing = required - set(instance)
        assert not missing, missing
        return
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    jsonschema.validate(instance=instance, schema=schema)


def test_schema_file_exists() -> None:
    assert SCHEMA.is_file()
    data = json.loads(SCHEMA.read_text(encoding="utf-8"))
    assert "blocked_missing_hazard_labels" in data["$defs"]["cellStatus"]["enum"]
    assert "skipped_tools_missing" in data["$defs"]["cellStatus"]["enum"]
    assert "xfail_known_issue" in data["$defs"]["cellStatus"]["enum"]


def test_valid_manifest_example() -> None:
    instance = {
        "schema_version": "1.0.0",
        "artifact_hash": "abc",
        "platform": "torch_ref",
        "cells": [
            {
                "condition_mode": "glaucoma",
                "platform": "torch_ref",
                "status": "blocked_missing_hazard_labels",
                "metrics": {},
                "gates_failed": ["SG-01", "SG-02"],
            }
        ],
        "all_passed": False,
        "summary": {"passed": 0, "failed": 0, "blocked": 1, "skipped": 0, "xfailed": 0},
    }
    _validate(instance)


def test_invalid_status_rejected_when_jsonschema_present() -> None:
    try:
        import jsonschema
    except ImportError:
        return
    instance = {
        "schema_version": "1.0.0",
        "artifact_hash": "abc",
        "platform": "torch_ref",
        "cells": [
            {
                "condition_mode": "glaucoma",
                "platform": "torch_ref",
                "status": "ok",
                "metrics": {},
                "gates_failed": [],
            }
        ],
        "all_passed": True,
        "summary": {"passed": 1, "failed": 0, "blocked": 0, "skipped": 0, "xfailed": 0},
    }
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    try:
        jsonschema.validate(instance=instance, schema=schema)
        raised = False
    except jsonschema.ValidationError:
        raised = True
    assert raised
