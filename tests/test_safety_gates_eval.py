"""Safety gate eval library tests (MAXS-301b)."""

from __future__ import annotations

import json
from pathlib import Path

from ml.evaluation.safety_gates import (
    STATUS_BLOCKED,
    STATUS_PASSED,
    build_certification_manifest,
    evaluate_condition_platform_cell,
)

GOLDEN = Path(__file__).resolve().parent / "fixtures" / "certification_manifest_example.json"


def test_missing_hazard_gt_fail_closed() -> None:
    cell = evaluate_condition_platform_cell(
        condition_mode="glaucoma",
        platform="torch_ref",
        hazard_ground_truth_available=False,
        metrics={
            "hazard_recall": 0.99,
            "false_safe_rate": 0.0,
            "direction_correctness": 0.99,
            "distance_zone_accuracy": 0.99,
        },
    )
    assert cell["status"] == STATUS_BLOCKED
    manifest = build_certification_manifest(
        artifact_hash="h", platform="torch_ref", cells=[cell]
    )
    assert manifest["all_passed"] is False


def test_all_passed_only_when_every_cell_passed() -> None:
    cell = evaluate_condition_platform_cell(
        condition_mode="glaucoma",
        platform="torch_ref",
        hazard_ground_truth_available=True,
        metrics={
            "hazard_recall": 0.99,
            "false_safe_rate": 0.0,
            "direction_correctness": 0.99,
            "distance_zone_accuracy": 0.99,
        },
    )
    assert cell["status"] == STATUS_PASSED
    manifest = build_certification_manifest(
        artifact_hash="h", platform="torch_ref", cells=[cell]
    )
    assert manifest["all_passed"] is True


def test_golden_fixture_shape() -> None:
    data = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert set(data) >= {
        "schema_version",
        "artifact_hash",
        "platform",
        "cells",
        "all_passed",
        "summary",
    }
    assert data["all_passed"] is False
