"""Local Stage A frame preprocessing. No network I/O."""

from __future__ import annotations

import numpy as np

from ml.runtime.stage_a.types import CameraFrame

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def frame_to_nchw_float(frame: CameraFrame, size: int = 224) -> np.ndarray:
    """Resize/normalize a CameraFrame to float32 NCHW [1, 3, H, W]."""
    image = frame.image
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    # Nearest-neighbor resize avoids a hard PIL dependency in the critical path.
    h, w = image.shape[:2]
    ys = (np.linspace(0, h - 1, size)).astype(np.int64)
    xs = (np.linspace(0, w - 1, size)).astype(np.int64)
    resized = image[ys][:, xs].astype(np.float32) / 255.0
    normalized = (resized - IMAGENET_MEAN) / IMAGENET_STD
    nchw = np.transpose(normalized, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(nchw, dtype=np.float32)
