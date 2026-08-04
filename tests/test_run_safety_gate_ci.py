"""Tests for run_safety_gate_ci thin wrapper (MAXS-301c)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "infra" / "run_safety_gate_ci.py"
GOLDEN_KEYS = {
    "schema_version",
    "artifact_hash",
    "platform",
    "cells",
    "all_passed",
    "summary",
}


def test_ci_writes_frozen_manifest_and_fails_closed(tmp_path: Path) -> None:
    out = tmp_path / "manifest.json"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(out), "--platform", "torch_ref"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1
    data = json.loads(out.read_text(encoding="utf-8"))
    assert GOLDEN_KEYS <= set(data)
    assert data["all_passed"] is False
    assert data["summary"]["blocked"] > 0
