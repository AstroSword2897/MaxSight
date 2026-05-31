"""Training reproducibility utilities and variance gates."""

from __future__ import annotations

import random
from typing import Any

import numpy as np


def set_deterministic_seed(seed: int) -> None:
    """Set Python, NumPy, and Torch seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def metric_variance(metrics_runs: list[dict[str, float]], key: str) -> float:
    """Return relative variance across runs for one metric key."""
    values = [float(run[key]) for run in metrics_runs if key in run]
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    if abs(mean) < 1e-12:
        return 0.0
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return (var**0.5) / abs(mean)


def assert_reproducible_metrics(
    metrics_runs: list[dict[str, float]],
    key: str = "val_loss",
    max_relative_variance: float = 0.01,
) -> None:
    """Raise when metric variance exceeds threshold (SCRUM-16 acceptance)."""
    rel = metric_variance(metrics_runs, key)
    if rel > max_relative_variance:
        raise RuntimeError(
            f"Metric {key} relative variance {rel:.4f} exceeds {max_relative_variance}"
        )


def reproducibility_manifest(
    *,
    seed: int,
    config_path: str | None = None,
    dataset_version: str | None = None,
    checkpoint_hash: str | None = None,
    config_hash: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build manifest attached to training artifacts."""
    manifest: dict[str, Any] = {
        "seed": seed,
        "config_path": config_path,
        "dataset_version": dataset_version,
        "deterministic_backends": True,
    }
    if checkpoint_hash is not None:
        manifest["checkpoint_hash"] = checkpoint_hash
    if config_hash is not None:
        manifest["config_hash"] = config_hash
    if extra:
        manifest.update(extra)
    return manifest


def checkpoint_content_hash(state_dict: dict[str, Any]) -> str:
    """Return a stable SHA256 hash over sorted state-dict tensor bytes."""
    import hashlib

    digest = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        digest.update(key.encode("utf-8"))
        value = state_dict[key]
        if hasattr(value, "detach"):
            digest.update(value.detach().cpu().numpy().tobytes())
        else:
            digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()
