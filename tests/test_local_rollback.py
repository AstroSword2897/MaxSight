"""Local rollback pointer-swap tests (MAXS-504)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.model_update import (
    ActivePointerWriteDenied,
    rollback_to_previous,
    staging_store,
)
from app.model_update.activation import promote_or_rollback


def test_rollback_writes_active_via_activation_only(tmp_path: Path) -> None:
    prev = tmp_path / "prev.bin"
    prev.write_bytes(b"prev")
    ptr = rollback_to_previous(tmp_path, prev)
    assert ptr.read_text(encoding="utf-8").strip() == str(prev)


def test_staging_store_still_denied_after_activation_api_exists(tmp_path: Path) -> None:
    store = staging_store(tmp_path)
    with pytest.raises(ActivePointerWriteDenied):
        store.write_active_pointer(tmp_path / "x")
    promote_or_rollback(tmp_path, tmp_path / "y.bin")
    assert staging_store(tmp_path).allow_active_writes is False
