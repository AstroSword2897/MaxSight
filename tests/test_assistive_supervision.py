"""Tests for deterministic assistive label derivation."""

from __future__ import annotations

import torch

from ml.data.assistive_supervision import (
    AssistiveSupervisionSpec,
    continuous_urgency_score,
    load_assistive_spec,
    object_distance_and_urgency,
    scene_urgency_from_objects,
    urgency_level_from_score,
)
from ml.training.assistive_metrics import AssistiveEvalAccumulator


def test_deterministic_object_labels():
    spec = load_assistive_spec()
    a = object_distance_and_urgency(0.5, 0.5, 0.3, 0.3, "car", spec)
    b = object_distance_and_urgency(0.5, 0.5, 0.3, 0.3, "car", spec)
    assert a == b


def test_car_near_center_higher_than_distant_person():
    spec = load_assistive_spec()
    car_score = continuous_urgency_score(0.5, 0.5, 0.25, 0.25, "car", spec)
    person_small = continuous_urgency_score(0.1, 0.1, 0.02, 0.02, "person", spec)
    assert car_score > person_small


def test_urgency_quantization_monotonic():
    spec = AssistiveSupervisionSpec(
        w_class=1.0,
        w_proximity=0.0,
        w_center=0.0,
        urgency_bin_edges=(0.25, 0.5, 0.75),
        distance_area_thresholds=(0.1, 0.05),
        default_class_prior=0.38,
        class_prior_overrides={},
    )
    assert urgency_level_from_score(0.1, spec) <= urgency_level_from_score(0.9, spec)


def test_assistive_eval_accumulator():
    acc = AssistiveEvalAccumulator()
    logits = torch.tensor([[0.0, 0.0, 4.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    gt = torch.tensor([3, 0])
    acc.update(logits, gt)
    m = acc.compute()
    assert m["hazard_recall_proxy"] == 1.0
    assert m["false_alert_rate_proxy"] == 0.0


def test_scene_urgency_max():
    assert scene_urgency_from_objects([1, 0, 2]) == 2
