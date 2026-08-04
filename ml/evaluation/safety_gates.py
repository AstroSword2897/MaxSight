"""Fail-closed safety-gate evaluation and frozen certification manifest shape."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ml.runtime_constants import (
    DIRECTION_CORRECTNESS_MIN,
    DISTANCE_ZONE_ACCURACY_MIN,
    FALSE_SAFE_RATE_MAX,
    HAZARD_RECALL_MIN,
    check_safety_gate_report,
)

SCHEMA_VERSION = "1.0.0"
DEFAULT_GATES_YAML = Path(__file__).resolve().parents[1] / "config" / "safety_gates.yaml"

STATUS_PASSED = "passed"
STATUS_FAILED = "failed"
STATUS_BLOCKED = "blocked_missing_hazard_labels"
STATUS_SKIPPED = "skipped_tools_missing"
STATUS_XFAIL = "xfail_known_issue"

STATUS_LABELS = {
    STATUS_PASSED: "PASS",
    STATUS_FAILED: "FAIL",
    STATUS_BLOCKED: "BLOCKED",
    STATUS_SKIPPED: "SKIP",
    STATUS_XFAIL: "XFAIL",
}


@dataclass(frozen=True)
class GateThresholds:
    hazard_recall_min: float = HAZARD_RECALL_MIN
    false_safe_rate_max: float = FALSE_SAFE_RATE_MAX
    direction_correctness_min: float = DIRECTION_CORRECTNESS_MIN
    distance_zone_accuracy_min: float = DISTANCE_ZONE_ACCURACY_MIN


def load_gate_thresholds(path: Path | None = None) -> GateThresholds:
    cfg_path = path or DEFAULT_GATES_YAML
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
    t = (data or {}).get("thresholds", {})
    return GateThresholds(
        hazard_recall_min=float(t.get("hazard_recall_min", HAZARD_RECALL_MIN)),
        false_safe_rate_max=float(t.get("false_safe_rate_max", FALSE_SAFE_RATE_MAX)),
        direction_correctness_min=float(
            t.get("direction_correctness_min", DIRECTION_CORRECTNESS_MIN)
        ),
        distance_zone_accuracy_min=float(
            t.get("distance_zone_accuracy_min", DISTANCE_ZONE_ACCURACY_MIN)
        ),
    )


def _summary_from_cells(cells: list[dict[str, Any]]) -> dict[str, int]:
    summary = {"passed": 0, "failed": 0, "blocked": 0, "skipped": 0, "xfailed": 0}
    for cell in cells:
        status = cell.get("status")
        if status == STATUS_PASSED:
            summary["passed"] += 1
        elif status == STATUS_FAILED:
            summary["failed"] += 1
        elif status == STATUS_BLOCKED:
            summary["blocked"] += 1
        elif status == STATUS_SKIPPED:
            summary["skipped"] += 1
        elif status == STATUS_XFAIL:
            summary["xfailed"] += 1
    return summary


def evaluate_condition_platform_cell(
    *,
    condition_mode: str,
    platform: str,
    metrics: dict[str, float] | None = None,
    hazard_ground_truth_available: bool = False,
    tools_available: bool = True,
    force_status: str | None = None,
) -> dict[str, Any]:
    """Evaluate one (condition, platform) cell. Skip/xfail never count as pass."""
    if force_status is not None:
        status = force_status
        gates_failed: list[str] = []
        metrics = metrics or {}
    elif not tools_available:
        status = STATUS_SKIPPED
        gates_failed = []
        metrics = metrics or {}
    elif not hazard_ground_truth_available:
        # Fail-closed: SG-01/02 cannot pass without hazard labels.
        status = STATUS_BLOCKED
        gates_failed = ["SG-01", "SG-02"]
        metrics = metrics or {}
    else:
        metrics = metrics or {}
        ok, failed = check_safety_gate_report(metrics)
        status = STATUS_PASSED if ok else STATUS_FAILED
        gates_failed = failed
    return {
        "condition_mode": condition_mode,
        "platform": platform,
        "status": status,
        "status_label": STATUS_LABELS.get(status, status.upper()),
        "metrics": metrics,
        "gates_failed": gates_failed,
    }


def build_certification_manifest(
    *,
    artifact_hash: str,
    platform: str,
    cells: list[dict[str, Any]],
    model_version: str = "",
) -> dict[str, Any]:
    """Frozen manifest shape consumed by CI/certify. Do not invent keys in callers."""
    summary = _summary_from_cells(cells)
    all_passed = bool(cells) and all(c.get("status") == STATUS_PASSED for c in cells)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_hash": artifact_hash,
        "platform": platform,
        "model_version": model_version,
        "cells": cells,
        "all_passed": all_passed,
        "summary": summary,
    }


def format_cell_status_line(cell: dict[str, Any]) -> str:
    """Human-readable line with explicit SKIP/XFAIL tokens."""
    label = cell.get("status_label") or STATUS_LABELS.get(cell.get("status", ""), "UNKNOWN")
    return (
        f"condition={cell.get('condition_mode')} platform={cell.get('platform')} "
        f"status={cell.get('status')} [{label}]"
    )
