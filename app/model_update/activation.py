"""Activation / rollback: sole ACTIVE_MODEL_PTR writer in First Wave."""

from __future__ import annotations

from pathlib import Path

from app.model_update.storage import activation_store


def promote_or_rollback(root: Path | str, target_artifact: Path | str) -> Path:
    """Set ACTIVE_MODEL_PTR to target (promotion or rollback). Local pointer swap only."""
    store = activation_store(root)
    return store.write_active_pointer(target_artifact)


def rollback_to_previous(root: Path | str, previous_artifact: Path | str) -> Path:
    return promote_or_rollback(root, previous_artifact)
