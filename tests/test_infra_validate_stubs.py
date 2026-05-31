"""Offline tests for scripts/infra/validate_infra_stubs.py."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_validate_infra_stubs_exits_zero():
    script = REPO / "scripts" / "infra" / "validate_infra_stubs.py"
    env = {**os.environ, "MAXSIGHT_INFRA_STRICT_PLACEHOLDERS": ""}
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout


def test_validate_infra_stubs_strict_fails_on_placeholder_iam():
    script = REPO / "scripts" / "infra" / "validate_infra_stubs.py"
    env = {**os.environ, "MAXSIGHT_INFRA_STRICT_PLACEHOLDERS": "1"}
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1, (
        "strict mode should fail while IAM stubs still contain {{placeholders}}"
    )
    combined = proc.stderr + proc.stdout
    assert "infra/iam/" in combined or "placeholders" in combined.lower()
