"""Detector adapter interface for Stage A (ticket 401, spec D3.3).

Every detector candidate (SSD-MobileNet first; challengers later) is wrapped in an adapter with the same
``detect(image, frame_id) -> DetectionBatch`` call. Candidates differ only in how they produce raw boxes,
scores and class indices. Everything after that is shared and fixed in ``finalize``:

* class map: the 17 classes of configs/data/label_map.yaml, in that order;
* score semantics: the model's per-class confidence in [0, 1], uncalibrated (calibration happens in the
  uncertainty assessor, D3.6); detections below ``score_threshold`` (PR-003, 0.35) are dropped;
* NMS: per class at IoU ``nms_iou`` (PR-004, 0.55);
* boxes: xyxy in source pixels, as schemas/detection-batch.schema.json requires, clipped to the image;
* order: highest score first, at most ``max_detections``.

The contract tests in tests/contract/test_detector_adapter_contract.py run against every adapter in ADAPTERS.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ml.datasets.label_map import LabelMap

SCHEMA_VERSION = "maxsight.stage_a/2.0"
Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class DetectorConfig:
    class_map: tuple[str, ...]
    score_threshold: float = 0.35  # PR-003 (experimental)
    nms_iou: float = 0.55  # PR-004 (experimental)
    max_detections: int = 64  # not in the parameter register; matches the tracker's 64-track table (D3)


DEFAULT_CONFIG = DetectorConfig(class_map=LabelMap.load().classes)


@dataclass(frozen=True)
class Detection:
    det_id: str
    class_name: str
    score: float
    bbox_xyxy: Box


@dataclass(frozen=True)
class DetectionBatch:
    frame_id: int
    model_version: str
    inference_ms: float
    detections: tuple[Detection, ...]
    skipped_reason: str | None = None

    def to_dict(self) -> dict:
        """The schema's JSON shape."""
        return {
            "schema_version": SCHEMA_VERSION,
            "frame_id": self.frame_id,
            "model_version": self.model_version,
            "inference_ms": self.inference_ms,
            "skipped_reason": self.skipped_reason,
            "detections": [
                {"det_id": d.det_id, "class_name": d.class_name, "score": d.score, "bbox_xyxy": list(d.bbox_xyxy)}
                for d in self.detections
            ],
        }


class DetectorAdapter(Protocol):
    config: DetectorConfig
    model_version: str

    def detect(self, image: np.ndarray, frame_id: int) -> DetectionBatch: ...


def box_iou(a: Box, b: Box) -> float:
    """Intersection over union of two xyxy boxes."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def finalize(
    boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, *, image_size: tuple[int, int],
    config: DetectorConfig, frame_id: int,
) -> tuple[Detection, ...]:
    """Shared post-processing from raw model outputs to detections.

    ``boxes`` is (N, 4) xyxy in source pixels, ``scores`` (N,) in [0, 1], ``classes`` (N,) indices into
    ``config.class_map``. ``image_size`` is (height, width).
    """
    if len(classes) and (classes.min() < 0 or classes.max() >= len(config.class_map)):
        raise ValueError(f"class index outside the class map (0..{len(config.class_map) - 1})")

    height, width = image_size
    clipped = np.column_stack([
        np.clip(boxes[:, 0], 0, width), np.clip(boxes[:, 1], 0, height),
        np.clip(boxes[:, 2], 0, width), np.clip(boxes[:, 3], 0, height),
    ]) if len(boxes) else np.zeros((0, 4))

    candidates = [
        (
            float(scores[i]),
            int(classes[i]),
            (
                float(clipped[i, 0]),
                float(clipped[i, 1]),
                float(clipped[i, 2]),
                float(clipped[i, 3]),
            ),
        )
        for i in range(len(scores))
        if scores[i] >= config.score_threshold and clipped[i, 2] > clipped[i, 0] and clipped[i, 3] > clipped[i, 1]
    ]
    candidates.sort(key=lambda c: -c[0])

    kept: list[tuple[float, int, Box]] = []
    for score, cls, box in candidates:  # greedy per-class NMS, highest score first
        if all(k_cls != cls or box_iou(box, k_box) <= config.nms_iou for _, k_cls, k_box in kept):
            kept.append((score, cls, box))
        if len(kept) == config.max_detections:
            break

    return tuple(
        Detection(det_id=f"d{frame_id}-{i}", class_name=config.class_map[cls], score=score, bbox_xyxy=box)
        for i, (score, cls, box) in enumerate(kept)
    )


class StaticDetector:
    """Reference adapter that returns fixed raw outputs, scaled to the input image.

    It exercises the contract and the shared post-processing without a trained model. It is not a
    detection candidate; real candidates (SSD-MobileNet, a challenger) are added to ADAPTERS when they exist.
    """

    model_version = "reference-static-1"

    # Normalised raw outputs: overlapping same-class boxes, a sub-threshold score and an out-of-frame box,
    # so the contract sees NMS, thresholding and clipping at work.
    _RAW = (
        ((0.10, 0.10, 0.40, 0.60), 0.92, "person"),
        ((0.12, 0.10, 0.42, 0.60), 0.80, "person"),
        ((0.50, 0.50, 0.70, 0.90), 0.66, "stairs"),
        ((0.80, 0.40, 1.20, 0.80), 0.51, "vehicle"),
        ((0.20, 0.70, 0.30, 0.95), 0.20, "door"),
    )

    def __init__(self, config: DetectorConfig = DEFAULT_CONFIG) -> None:
        self.config = config

    def detect(self, image: np.ndarray, frame_id: int) -> DetectionBatch:
        start = time.perf_counter()
        height, width = image.shape[:2]
        scale = np.array([width, height, width, height], dtype=float)
        boxes = np.array([np.array(b) * scale for b, _, _ in self._RAW])
        scores = np.array([s for _, s, _ in self._RAW])
        classes = np.array([self.config.class_map.index(c) for _, _, c in self._RAW])
        detections = finalize(boxes, scores, classes, image_size=(height, width), config=self.config,
                              frame_id=frame_id)
        return DetectionBatch(frame_id=frame_id, model_version=self.model_version,
                              inference_ms=(time.perf_counter() - start) * 1000.0, detections=detections)


# Every adapter here must pass tests/contract/test_detector_adapter_contract.py unchanged.
ADAPTERS: dict[str, Callable[[], DetectorAdapter]] = {
    "reference-static": StaticDetector,
}
