"""OTA staging storage-layer ACTIVE deny tests (MAXS-501)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.model_update import ActivePointerWriteDenied, stage_candidate, staging_store


def test_staging_cannot_write_active_pointer(tmp_path: Path) -> None:
    store = staging_store(tmp_path)
    with pytest.raises(ActivePointerWriteDenied):
        store.write_active_pointer(tmp_path / "x.bin")
    with pytest.raises(ActivePointerWriteDenied):
        store.write_staging("ACTIVE_MODEL_PTR", b"evil")


def test_stage_candidate_leaves_active_untouched(tmp_path: Path) -> None:
    store = staging_store(tmp_path)
    artifact = tmp_path / "cand.bin"
    artifact.write_bytes(b"weights")
    manifest = {
        "all_passed": True,
        "cells": [{"status": "passed"}],
        "summary": {"passed": 1, "failed": 0, "blocked": 0, "skipped": 0, "xfailed": 0},
    }
    staged = stage_candidate(store=store, artifact_path=artifact, manifest=manifest)
    assert staged.is_file()
    assert store.read_active_pointer() is None
    assert not store.active_path.exists()
