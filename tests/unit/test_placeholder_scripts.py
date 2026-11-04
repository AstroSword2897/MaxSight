"""Unimplemented entry points must fail loudly and never pretend success."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _run(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(REPO / "scripts" / script), *args],  # noqa: S603
                          capture_output=True, text=True, timeout=60, check=False)



@pytest.mark.parametrize("script", ["train_detector.py", "evaluate_detector.py", "export_model.py",
                                    "generate_golden_vectors.py", "validate_release.py"])
def test_unimplemented_scripts_fail_loudly_never_pretend_success(script):
    out = _run(script)
    assert out.returncode != 0 and "not implemented" in out.stderr
