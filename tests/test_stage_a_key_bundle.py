"""Tests for key bundle presence (MAXS-203a)."""

from __future__ import annotations

from pathlib import Path

KEYS = Path(__file__).resolve().parents[1] / "ml" / "runtime" / "stage_a" / "keys"


def test_trust_window_files_present() -> None:
    assert (KEYS / "current.pub").is_file()
    assert (KEYS / "next.pub").is_file()
    assert (KEYS / "current.hmac").is_file()
    assert (KEYS / "next.hmac").is_file()
    assert (KEYS / "README.md").is_file()
