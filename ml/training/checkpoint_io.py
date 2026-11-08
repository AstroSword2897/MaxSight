"""Checkpoint file I/O: atomic writes and a strict reader (ticket 402).

Kept free of the training loop's heavy imports so it can be tested and reused on its own.
The reader never continues silently: a missing file, a corrupt file and a file without model weights
all raise.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import torch

REQUIRED_KEY = "model_state_dict"


class CheckpointError(RuntimeError):
    """A checkpoint file exists but cannot be used (corrupt, truncated or missing weights)."""


def write_atomic_torch(path: Path, payload: Any) -> None:
    """Write a torch checkpoint atomically so a crash mid-write never corrupts the file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _load_weights_only(path: Path, map_location: str | torch.device) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    try:
        return torch.load(str(path), map_location=map_location, weights_only=True)
    except Exception as exc:  # torch raises several types for unreadable files
        raise CheckpointError(f"checkpoint is corrupt or unreadable: {path} ({type(exc).__name__}: {exc})") from exc


def read_checkpoint(path: Path | str, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a checkpoint saved by write_atomic_torch.

    Raises FileNotFoundError if the file is missing, and CheckpointError if it cannot be unpickled
    with ``weights_only=True`` or does not contain ``model_state_dict``.
    """
    path = Path(path)
    checkpoint = _load_weights_only(path, map_location)
    if not isinstance(checkpoint, dict) or REQUIRED_KEY not in checkpoint:
        raise CheckpointError(f"checkpoint has no {REQUIRED_KEY}: {path}")
    return checkpoint


def _state_dict_of(payload: Any, path: Path) -> dict[str, torch.Tensor]:
    """Accept ``{"model_state_dict": ...}`` or a bare state dict; anything else is not a weights file."""
    state = payload.get(REQUIRED_KEY, payload) if isinstance(payload, dict) else None
    if not isinstance(state, dict) or not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise CheckpointError(f"checkpoint holds no model weights: {path}")
    return state


def _mismatches(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> list[str]:
    expected = model.state_dict()
    problems = [f"missing {k}" for k in expected if k not in state]
    problems += [f"unexpected {k}" for k in state if k not in expected]
    problems += [
        f"shape {k}: checkpoint {tuple(state[k].shape)} vs model {tuple(v.shape)}"
        for k, v in expected.items()
        if k in state and state[k].shape != v.shape
    ]
    return problems


def load_model_weights(
    model: torch.nn.Module, path: Path | str, map_location: str | torch.device = "cpu"
) -> torch.nn.Module:
    """Load weights for inference or export, all or nothing.

    Every model key must be present with the same shape, and the checkpoint may not carry extra keys.
    The check runs before any tensor is copied, so a refused checkpoint leaves the model untouched
    (``nn.Module.load_state_dict`` copies matching tensors before it reports errors).
    Raises FileNotFoundError or CheckpointError; never falls back to random initialisation.
    """
    path = Path(path)
    state = _state_dict_of(_load_weights_only(path, map_location), path)
    problems = _mismatches(model, state)
    if problems:
        shown = "; ".join(problems[:10]) + (f"; ... {len(problems) - 10} more" if len(problems) > 10 else "")
        raise CheckpointError(f"checkpoint does not match the model: {path} ({shown})")
    model.load_state_dict(state, strict=True)
    return model
