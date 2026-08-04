"""Per-platform certification with explicit SKIP/XFAIL labels (MAXS-303)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from ml.evaluation.safety_gates import (
    STATUS_SKIPPED,
    STATUS_XFAIL,
    build_certification_manifest,
    evaluate_condition_platform_cell,
    format_cell_status_line,
)

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "infra" / "run_safety_gate_ci.py"


def test_skip_never_counts_as_pass() -> None:
    cell = evaluate_condition_platform_cell(
        condition_mode="glaucoma",
        platform="onnx",
        tools_available=False,
    )
    assert cell["status"] == STATUS_SKIPPED
    assert cell["status_label"] == "SKIP"
    line = format_cell_status_line(cell)
    assert "[SKIP]" in line
    manifest = build_certification_manifest(
        artifact_hash="h", platform="onnx", cells=[cell]
    )
    assert manifest["all_passed"] is False
    assert manifest["summary"]["skipped"] == 1
    assert manifest["summary"]["passed"] == 0


def test_xfail_never_counts_as_pass() -> None:
    cell = evaluate_condition_platform_cell(
        condition_mode="glaucoma",
        platform="coreml",
        force_status=STATUS_XFAIL,
    )
    assert cell["status_label"] == "XFAIL"
    assert "[XFAIL]" in format_cell_status_line(cell)
    manifest = build_certification_manifest(
        artifact_hash="h", platform="coreml", cells=[cell]
    )
    assert manifest["all_passed"] is False
    assert manifest["summary"]["xfailed"] == 1


def test_cli_prints_skip_token(tmp_path: Path) -> None:
    out = tmp_path / "m.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output",
            str(out),
            "--platform",
            "coreml",
            "--tools-missing",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1
    assert "[SKIP]" in proc.stdout
    assert "skipped=" in proc.stdout
