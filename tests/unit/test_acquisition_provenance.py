"""Tests for acquisition provenance. None of these tests use the network.

They cover one guarantee so far:
  1. Status names say exactly how far a dataset has been checked.
"""

from __future__ import annotations

import pytest
from ml.datasets.sources import (
    VALIDATED_MARKER,
    State,
    load_registry,
    status_label,
)

REGISTRY = load_registry()


# ----------------------------------------------------------------------------------------------
# 1. Status vocabulary
# ----------------------------------------------------------------------------------------------

def test_state_names_say_exactly_what_was_verified():
    """The ladder uses explicit names, never a vague 'COMPLETE'."""
    names = [state.name for state in State]

    assert names == [
        "NOT_DOWNLOADED",
        "DOWNLOADED_UNVERIFIED",
        "RAW_VERIFIED",
        "PREPROCESSED_UNVERIFIED",
        "VALIDATED",
    ]


@pytest.mark.parametrize(
    ("source_id", "expected_label"),
    [
        ("coco2017", "READY_FOR_DETECTOR_PRETRAINING"),
        ("bdd100k", "READY_FOR_EVALUATION_ONLY"),
        ("diode", "READY_FOR_RESEARCH_EXPERIMENTS"),
    ],
)
def test_validated_label_names_the_purpose_from_the_role(tmp_path, source_id, expected_label):
    """A validated dataset is only ready for the purpose its registry role allows."""
    # Arrange
    source = REGISTRY[source_id]
    assert status_label(source, tmp_path) == "NOT_DOWNLOADED"

    validated_marker = tmp_path / "processed" / source_id / VALIDATED_MARKER
    validated_marker.parent.mkdir(parents=True)
    validated_marker.write_text("{}")

    # Act
    label = status_label(source, tmp_path)

    # Assert
    assert label == expected_label
