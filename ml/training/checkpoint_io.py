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


def read_checkpoint(path: Path | str, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a checkpoint saved by write_atomic_torch.

    Raises FileNotFoundError if the file is missing, and CheckpointError if it cannot be unpickled
    with ``weights_only=True`` or does not contain ``model_state_dict``.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    try:
        checkpoint = torch.load(str(path), map_location=map_location, weights_only=True)
    except Exception as exc:  # torch raises several types for unreadable files
        raise CheckpointError(f"checkpoint is corrupt or unreadable: {path} ({type(exc).__name__}: {exc})") from exc
    if not isinstance(checkpoint, dict) or REQUIRED_KEY not in checkpoint:
        raise CheckpointError(f"checkpoint has no {REQUIRED_KEY}: {path}")
    return checkpoint
