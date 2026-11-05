"""Ticket 401: every detector adapter passes the same contract.

The suite is parametrised over ``ml.runtime.stage_a.detector.ADAPTERS``. A new candidate (SSD-MobileNet, a
challenger) is added to that registry and must pass every test here unchanged.

Contract (spec D3.3, PR-003, PR-004):
  * output is a DetectionBatch that validates against schemas/detection-batch.schema.json;
  * boxes are xyxy in source pixels, inside the image, with x1 < x2 and y1 < y2;
  * class names come from the fixed class map (configs/data/label_map.yaml);
  * scores are in [score_threshold, 1] and sorted high to low;
  * per-class NMS at the fixed IoU leaves no same-class pair above it;
  * at most max_detections detections; the same input gives the same output.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jsonschema")
pytest.importorskip("referencing")

from ml.datasets.label_map import LabelMap  # noqa: E402
from ml.runtime.stage_a.detector import ADAPTERS, DEFAULT_CONFIG, box_iou  # noqa: E402

IMAGE = np.zeros((240, 320, 3), dtype=np.uint8)  # height 240, width 320


@pytest.fixture(params=sorted(ADAPTERS), ids=lambda name: name)
def adapter(request):
    return ADAPTERS[request.param]()


def run(adapter, image=IMAGE, frame_id: int = 7):
    return adapter.detect(image, frame_id=frame_id)


def test_output_validates_against_the_schema(adapter, validator):
    validator("detection-batch").validate(run(adapter).to_dict())


def test_frame_and_model_are_identified(adapter):
    batch = run(adapter, frame_id=42)
    assert batch.frame_id == 42
    assert batch.model_version
    assert batch.inference_ms >= 0


def test_boxes_are_source_pixel_xyxy_inside_the_image(adapter):
    height, width = IMAGE.shape[:2]
    for det in run(adapter).detections:
        x1, y1, x2, y2 = det.bbox_xyxy
        assert 0 <= x1 < x2 <= width, det
        assert 0 <= y1 < y2 <= height, det


def test_class_names_come_from_the_fixed_class_map(adapter):
    classes = set(LabelMap.load().classes)
    assert adapter.config.class_map == LabelMap.load().classes
    assert {det.class_name for det in run(adapter).detections} <= classes


def test_scores_respect_the_threshold_and_are_sorted(adapter):
    scores = [det.score for det in run(adapter).detections]
    assert all(adapter.config.score_threshold <= s <= 1.0 for s in scores)
    assert scores == sorted(scores, reverse=True)


def test_no_same_class_pair_overlaps_above_the_nms_iou(adapter):
    detections = run(adapter).detections
    for i, a in enumerate(detections):
        for b in detections[i + 1 :]:
            if a.class_name == b.class_name:
                assert box_iou(a.bbox_xyxy, b.bbox_xyxy) <= adapter.config.nms_iou, (a, b)


def test_detection_count_is_capped(adapter):
    assert len(run(adapter).detections) <= adapter.config.max_detections


def test_the_same_input_gives_the_same_output(adapter):
    first, second = run(adapter), run(adapter)
    assert first.detections == second.detections


def test_det_ids_are_unique_within_a_batch(adapter):
    ids = [det.det_id for det in run(adapter).detections]
    assert len(ids) == len(set(ids))


def test_every_adapter_uses_the_shared_fixed_settings(adapter):
    """Fixed class map, score semantics and NMS settings: candidates differ only in the model."""
    assert adapter.config == DEFAULT_CONFIG
