"""Shared detector post-processing (ticket 401): thresholding, per-class NMS, clipping, ordering, capping."""

from __future__ import annotations

import numpy as np
import pytest
from ml.runtime.stage_a.detector import DEFAULT_CONFIG, DetectorConfig, box_iou, finalize


def raw(boxes, scores, classes):
    return np.array(boxes, dtype=float), np.array(scores, dtype=float), np.array(classes, dtype=int)


def test_box_iou():
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert box_iou((0, 0, 10, 10), (10, 10, 20, 20)) == 0.0
    assert box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_scores_below_the_threshold_are_dropped():
    boxes, scores, classes = raw([[0, 0, 10, 10], [20, 20, 30, 30]], [0.34, 0.35], [0, 0])
    kept = finalize(boxes, scores, classes, image_size=(100, 100), config=DEFAULT_CONFIG, frame_id=1)
    assert [d.score for d in kept] == [0.35]


def test_nms_is_per_class():
    # Two overlapping boxes of the same class: one survives. Same boxes, different classes: both survive.
    boxes, scores, classes = raw([[0, 0, 10, 10], [1, 0, 11, 10], [1, 0, 11, 10]], [0.9, 0.8, 0.7], [0, 0, 1])
    kept = finalize(boxes, scores, classes, image_size=(100, 100), config=DEFAULT_CONFIG, frame_id=1)
    assert [(d.class_name, d.score) for d in kept] == [
        (DEFAULT_CONFIG.class_map[0], 0.9), (DEFAULT_CONFIG.class_map[1], 0.7)]


def test_boxes_are_clipped_to_the_image_and_degenerate_boxes_dropped():
    boxes, scores, classes = raw([[-5, -5, 50, 40], [95, 95, 120, 130], [60, 60, 60, 70]], [0.9, 0.8, 0.7], [0, 1, 2])
    kept = finalize(boxes, scores, classes, image_size=(100, 100), config=DEFAULT_CONFIG, frame_id=1)
    assert [d.bbox_xyxy for d in kept] == [(0.0, 0.0, 50.0, 40.0), (95.0, 95.0, 100.0, 100.0)]


def test_output_is_sorted_and_capped():
    config = DetectorConfig(class_map=DEFAULT_CONFIG.class_map, score_threshold=0.0, nms_iou=0.55, max_detections=2)
    boxes, scores, classes = raw([[0, 0, 5, 5], [10, 10, 15, 15], [20, 20, 25, 25]], [0.2, 0.9, 0.5], [0, 0, 0])
    kept = finalize(boxes, scores, classes, image_size=(100, 100), config=config, frame_id=3)
    assert [d.score for d in kept] == [0.9, 0.5]
    assert [d.det_id for d in kept] == ["d3-0", "d3-1"]


def test_an_unknown_class_index_is_an_error():
    boxes, scores, classes = raw([[0, 0, 10, 10]], [0.9], [len(DEFAULT_CONFIG.class_map)])
    with pytest.raises(ValueError, match="class index"):
        finalize(boxes, scores, classes, image_size=(100, 100), config=DEFAULT_CONFIG, frame_id=1)


def test_default_settings_match_the_parameter_register():
    assert DEFAULT_CONFIG.score_threshold == 0.35  # PR-003
    assert DEFAULT_CONFIG.nms_iou == 0.55  # PR-004
