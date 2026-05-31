#!/usr/bin/env python3
"""Validate runtime API contracts, OpenAPI, and JSON schemas."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ml.runtime.contracts import (  # noqa: E402
    CriticalEvent,
    RuntimeResponse,
    validate_runtime_response,
)


def _errors() -> list[str]:
    errors: list[str] = []
    openapi = REPO / "docs/contracts/openapi.yaml"
    schema = REPO / "docs/contracts/schemas/runtime_response.json"
    deprecations = REPO / "scripts/product/deprecations.yaml"

    if not openapi.exists():
        errors.append("Missing docs/contracts/openapi.yaml")
    if not schema.exists():
        errors.append("Missing docs/contracts/schemas/runtime_response.json")
    if not deprecations.exists():
        errors.append("Missing scripts/product/deprecations.yaml")

    try:
        schema_data = json.loads(schema.read_text(encoding="utf-8"))
        required = set(schema_data.get("required", []))
        expected = {
            "frame_id",
            "tier",
            "degraded_mode",
            "classifications",
            "critical_events",
            "secondary_events",
            "therapy",
            "latency_ms",
        }
        if not expected.issubset(required):
            errors.append(f"Schema missing required fields: {sorted(expected - required)}")
    except Exception as exc:
        errors.append(f"Schema parse failed: {exc}")

    sample = RuntimeResponse(
        frame_id="f1",
        tier=__import__("ml.runtime.contracts", fromlist=["ComputeTier"]).ComputeTier.SILVER,
        degraded_mode=__import__(
            "ml.runtime.contracts", fromlist=["DegradedMode"]
        ).DegradedMode.D0_NORMAL,
        critical_events=[
            CriticalEvent(
                event_type="obstacle",
                urgency=2,
                direction="center",
                distance_zone="near",
                confidence=0.9,
                uncertainty=0.1,
                timestamp_source=1.0,
                timestamp_emit=1.05,
            )
        ],
    )
    try:
        validate_runtime_response(sample.to_dict())
    except ValueError as exc:
        errors.append(f"Sample runtime response invalid: {exc}")

    ownership = REPO / "docs/architecture/module_ownership.md"
    if not ownership.exists():
        errors.append("Missing docs/architecture/module_ownership.md")

    return errors


def main() -> int:
    errors = _errors()
    if errors:
        for line in errors:
            print(line, file=sys.stderr)
        return 1
    print("OK: runtime contracts validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
