"""Python Stage A types must agree with the JSON schemas (spec D3.3: schemas are the source of truth).

The legacy ``ml.runtime.stage_a.types`` predates the schemas, so these are strict xfails that Step C4
of the restructure plan must flip to passes.
"""

from __future__ import annotations

import dataclasses

import pytest

types = pytest.importorskip("ml.runtime.stage_a.types")


def _fields(cls) -> dict[str, str]:
    return {f.name: str(f.type) for f in dataclasses.fields(cls)}


def test_camera_frame_keeps_image_and_timestamp():
    assert {"image", "timestamp"} <= set(_fields(types.CameraFrame))


@pytest.mark.xfail(strict=True, reason="schema frame_id is an integer >= 1; legacy CameraFrame uses str")
def test_camera_frame_id_is_integer():
    assert _fields(types.CameraFrame)["frame_id"] == "int"


@pytest.mark.xfail(strict=True, reason="perception is condition-independent; HazardResult still has condition_mode")
def test_hazard_result_has_no_condition_field():
    assert "condition_mode" not in _fields(types.HazardResult)


@pytest.mark.xfail(strict=True, reason="schema urgency is 0..3 with a separate system-health state; "
                                        "HazardResult has no frame_id/hazard_id/track_id linkage")
def test_hazard_result_links_to_frame_and_track():
    assert {"frame_id", "track_id"} <= set(_fields(types.HazardResult))


def test_hazard_result_distance_is_optional_never_defaulted_to_zero():
    result_fields = {f.name: f for f in dataclasses.fields(types.HazardResult)}
    assert result_fields["distance_meters"].default is None
