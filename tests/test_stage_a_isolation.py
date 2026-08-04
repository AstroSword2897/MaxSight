"""Tests for Stage A import isolation validator (MAXS-101b)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "infra" / "validate_stage_a_isolation.py"


def test_isolation_validator_passes_on_clean_tree() -> None:
    code = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert code.returncode == 0, code.stderr


def test_isolation_self_test_detects_violation() -> None:
    code = subprocess.run(
        [sys.executable, str(SCRIPT), "--self-test"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert code.returncode == 0, code.stderr
    assert "self-test OK" in code.stdout
