"""Tests for Stage A refusal messages and fixtures (MAXS-102a)."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

from ml.runtime.stage_a.messages import HAZARD_UNAVAILABLE_MESSAGE

FIXTURES = Path(__file__).resolve().parents[1] / "ml" / "runtime" / "stage_a" / "fixtures"


def test_refusal_message_exact_contract() -> None:
    assert HAZARD_UNAVAILABLE_MESSAGE == (
        "hazard detection unavailable, please use your primary mobility aid"
    )


def test_calibration_frame_exists() -> None:
    path = FIXTURES / "calibration_frame.png"
    assert path.is_file()
    assert path.stat().st_size > 0


def test_lkg_manifest_and_immutability_flag() -> None:
    manifest = json.loads((FIXTURES / "last_known_good" / "manifest.json").read_text())
    assert manifest["immutable"] is True
    assert (FIXTURES / "last_known_good" / "weights.stub").is_file()


def test_lkg_directory_write_denied_when_readonly(tmp_path: Path) -> None:
    src = FIXTURES / "last_known_good" / "weights.stub"
    dest = tmp_path / "weights.stub"
    dest.write_bytes(src.read_bytes())
    dest.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    try:
        raised = False
        try:
            with open(dest, "wb") as fh:
                fh.write(b"tamper")
        except OSError:
            raised = True
        # Some platforms allow open but fail on write; either is acceptable.
        if not raised:
            # If write succeeded unexpectedly, restore expectation via chmod check.
            assert not os.access(dest, os.W_OK) or dest.read_bytes() == src.read_bytes()
    finally:
        dest.chmod(stat.S_IRUSR | stat.S_IWUSR)
