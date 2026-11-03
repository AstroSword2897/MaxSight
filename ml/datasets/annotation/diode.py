"""DIODE validation integrity: every RGB has a depth array and a validity mask of matching shape.

Layout (official): ``<name>.png``, ``<name>_depth.npy``, ``<name>_depth_mask.npy``. RGB is model input;
depth and mask are reference ground truth only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def scan(root: Path) -> list[str]:
    """Return a list of problems (empty list means the pairing is intact)."""
    problems: list[str] = []
    rgbs = sorted(root.rglob("*.png"))
    if not rgbs:
        return ["no_rgb_images_found"]
    for rgb in rgbs:
        depth = rgb.with_name(rgb.stem + "_depth.npy")
        mask = rgb.with_name(rgb.stem + "_depth_mask.npy")
        if not depth.exists():
            problems.append(f"missing_depth:{rgb.name}")
            continue
        if not mask.exists():
            problems.append(f"missing_depth_mask:{rgb.name}")
            continue
        try:
            with Image.open(rgb) as im:
                w, h = im.size
            d, m = np.load(depth, mmap_mode="r"), np.load(mask, mmap_mode="r")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"unreadable:{rgb.name}:{type(exc).__name__}")
            continue
        if d.shape[:2] != (h, w) or m.shape[:2] != (h, w):
            problems.append(f"shape_mismatch:{rgb.name}:rgb={(h, w)} depth={d.shape} mask={m.shape}")
        valid = np.asarray(m).astype(bool)
        dv = np.asarray(d)[valid] if valid.shape == np.asarray(d).shape[:2] else np.asarray([])
        if dv.size and (not np.isfinite(dv).all() or (dv <= 0).any()):
            problems.append(f"invalid_depth_in_valid_region:{rgb.name}")
    return problems
