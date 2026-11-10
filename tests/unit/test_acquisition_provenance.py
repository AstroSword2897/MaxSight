"""Tests for acquisition provenance. None of these tests use the network.

They cover two guarantees so far:
  1. Status names say exactly how far a dataset has been checked.
  2. The ACQUIRED marker keeps one entry per split, so a second split never erases the first.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ml.datasets.acquisition import coco
from ml.datasets.sources import (
    ACQUIRED_MARKER,
    VALIDATED_MARKER,
    State,
    current_state,
    load_registry,
    status_label,
    write_acquired_marker,
)

REGISTRY = load_registry()


# ----------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------

def read_marker(raw_dir: Path) -> dict:
    return json.loads((raw_dir / ACQUIRED_MARKER).read_text())


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


# ----------------------------------------------------------------------------------------------
# 2. Per-split ACQUIRED marker
# ----------------------------------------------------------------------------------------------

def test_marker_keeps_every_split_instead_of_overwriting(tmp_path):
    """Acquiring val2017 after train2017 must not erase the train2017 record."""
    # Arrange
    raw_dir = tmp_path / "raw" / "coco2017"

    # Act
    write_acquired_marker(raw_dir, verified=False, split="train2017", images=10)
    write_acquired_marker(raw_dir, verified=False, split="val2017", images=4)

    # Assert
    marker = read_marker(raw_dir)
    assert marker["splits"]["train2017"]["images"] == 10
    assert marker["splits"]["val2017"]["images"] == 4
    assert marker["verified"] is False


def test_marker_is_verified_only_when_every_split_is(tmp_path):
    """One unverified split keeps the whole dataset unverified."""
    raw_dir = tmp_path / "raw" / "coco2017"

    # One split verified, one not: the dataset is not verified.
    write_acquired_marker(raw_dir, verified=True, split="val2017", images=4)
    write_acquired_marker(raw_dir, verified=False, split="train2017", images=10)
    assert read_marker(raw_dir)["verified"] is False

    # Both splits verified: now it is.
    write_acquired_marker(raw_dir, verified=True, split="train2017", images=10)
    assert read_marker(raw_dir)["verified"] is True
    assert current_state(REGISTRY["coco2017"], tmp_path) is State.RAW_VERIFIED


def test_legacy_single_split_marker_is_folded_into_splits(tmp_path):
    """The Open Images marker written before this change (one top-level split) is preserved."""
    # Arrange: the old format, exactly as it exists on disk today.
    raw_dir = tmp_path / "raw" / "open_images_v7"
    raw_dir.mkdir(parents=True)
    legacy_marker = {
        "verified": False,
        "acquired_at": "2026-11-03T19:39:50Z",
        "split": "validation",
        "images": 6278,
    }
    (raw_dir / ACQUIRED_MARKER).write_text(json.dumps(legacy_marker))

    # Act
    write_acquired_marker(raw_dir, verified=False, split="test", images=5)

    # Assert: the old split survives with its original timestamp, and the new split is added.
    splits = read_marker(raw_dir)["splits"]
    assert splits["validation"]["images"] == 6278
    assert splits["validation"]["acquired_at"] == "2026-11-03T19:39:50Z"
    assert splits["test"]["images"] == 5


def test_coco_manifest_path_is_per_split(tmp_path):
    """train2017 and val2017 must not share one manifest file."""
    train_manifest = coco.manifest_path(tmp_path, "train2017")
    val_manifest = coco.manifest_path(tmp_path, "val2017")

    assert train_manifest != val_manifest
