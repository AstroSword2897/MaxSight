"""Leakage-safe splits: whole groups (clip / session / environment) go to exactly one split."""

from __future__ import annotations

import hashlib

from ml.datasets.manifest import ImageRecord


# Splits that mean "never train on this" and are set by the source converter (e.g. BDD100K is eval-only).
PRESERVED_SPLITS = frozenset({"eval"})


def _bucket(group_id: str, seed: int) -> float:
    h = hashlib.sha256(f"{seed}:{group_id}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def assign_splits(
    records: list[ImageRecord],
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    holdout_groups: frozenset[str] = frozenset(),
    seed: int = 0,
) -> list[ImageRecord]:
    """Deterministic group split. ``holdout_groups`` (e.g. a whole environment) become ``test_holdout``.

    Records already tagged with a preserved split (``eval``) are returned unchanged.
    """
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("ratios must sum to 1")
    train_hi, val_hi = ratios[0], ratios[0] + ratios[1]
    out = []
    for rec in records:
        if rec.split in PRESERVED_SPLITS:
            out.append(rec)
            continue
        if rec.group_id in holdout_groups or rec.extra.get("environment") in holdout_groups:
            split = "test_holdout"
        else:
            x = _bucket(rec.group_id, seed)
            split = "train" if x < train_hi else "val" if x < val_hi else "test"
        out.append(rec.with_(split=split))
    return out
