"""Tests for acquisition provenance. None of these tests use the network.

They cover four guarantees:
  1. Status names say exactly how far a dataset has been checked.
  2. The ACQUIRED marker keeps one entry per split, so a second split never erases the first.
  3. Every downloaded image has its SHA-256 recorded in the manifest.
  4. Raw verification fails loudly on missing, corrupt or changed files.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest
from ml.datasets.acquisition import coco
from ml.datasets.acquisition import openimages_manual as oim
from ml.datasets.integrity import verify_raw
from ml.datasets.label_map import LabelMap
from ml.datasets.manifest import BoxRecord, ImageRecord, write_manifest
from ml.datasets.sources import (
    ACQUIRED_MARKER,
    VALIDATED_MARKER,
    State,
    current_state,
    load_registry,
    status_label,
    write_acquired_marker,
)
from PIL import Image

REGISTRY = load_registry()


# ----------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------

def make_jpeg_bytes(width: int = 8, height: int = 6) -> bytes:
    """Return the bytes of a small, valid JPEG image."""
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color=(10, 20, 30)).save(buffer, format="JPEG")
    return buffer.getvalue()


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_record(relative_path: str, sha256: str = "") -> ImageRecord:
    """A one-box COCO manifest record pointing at ``relative_path``."""
    person_box = BoxRecord(cls="person", bbox_xyxy=(0, 0, 4, 4), source_class="person")
    return ImageRecord(
        image_id=f"coco:{relative_path}",
        source="coco",
        path=relative_path,
        width=8,
        height=6,
        group_id=f"coco:{relative_path}",
        boxes=(person_box,),
        sha256=sha256,
    )


def read_marker(raw_dir: Path) -> dict:
    return json.loads((raw_dir / ACQUIRED_MARKER).read_text())


class FakeHttpResponse(io.BytesIO):
    """Stands in for the object urllib.request.urlopen returns (a context manager with .read())."""

    def __enter__(self) -> FakeHttpResponse:
        return self

    def __exit__(self, *exc_info) -> bool:
        return False


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


# ----------------------------------------------------------------------------------------------
# 3. Per-image SHA-256
# ----------------------------------------------------------------------------------------------

def test_coco_download_records_the_sha256_of_the_bytes_written(tmp_path, monkeypatch):
    """A freshly downloaded image is hashed from the bytes that were written."""
    # Arrange: replace the network call with a fake response.
    image_bytes = make_jpeg_bytes()

    def fake_urlopen(request, timeout=30):
        return FakeHttpResponse(image_bytes)

    monkeypatch.setattr(coco.urllib.request, "urlopen", fake_urlopen)
    record = make_record("images/a.jpg").with_(url="http://example.invalid/a.jpg")

    # Act
    downloaded, size, error = coco._download_one(record, tmp_path)

    # Assert
    assert error == ""
    assert size == len(image_bytes)
    assert downloaded.sha256 == sha256_of(image_bytes)


def test_coco_download_hashes_a_file_that_was_already_on_disk(tmp_path):
    """A resumed run skips the download but still records the hash of the existing file."""
    # Arrange
    image_bytes = make_jpeg_bytes()
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "a.jpg").write_bytes(image_bytes)

    # Act
    downloaded, _size, error = coco._download_one(make_record("images/a.jpg"), tmp_path)

    # Assert
    assert error == ""
    assert downloaded.sha256 == sha256_of(image_bytes)


def test_coco_manifest_path_is_per_split(tmp_path):
    """train2017 and val2017 must not share one manifest file."""
    train_manifest = coco.manifest_path(tmp_path, "train2017")
    val_manifest = coco.manifest_path(tmp_path, "val2017")

    assert train_manifest != val_manifest


def test_openimages_records_carry_the_file_sha256(tmp_path):
    """Open Images manifest records include the SHA-256 of the image on disk."""
    # Arrange: one downloaded image with one Door box.
    image_bytes = make_jpeg_bytes()
    image_path = tmp_path / "images" / "validation" / "abc.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(image_bytes)

    door_box = oim.BoxRow(
        label_id="/m/door",
        xmin=0.1, xmax=0.5, ymin=0.1, ymax=0.5,
        is_group_of=False, is_depiction=False,
    )
    selection = oim.Selection(images={"abc": [door_box]}, class_counts={})

    # Act
    records = oim.to_records(selection, {"/m/door": "Door"}, "validation", tmp_path, LabelMap.load())

    # Assert
    assert len(records) == 1
    assert records[0].sha256 == sha256_of(image_bytes)


# ----------------------------------------------------------------------------------------------
# 4. Raw verification
# ----------------------------------------------------------------------------------------------

def test_verify_raw_passes_only_complete_decodable_matching_files(tmp_path):
    """Each kind of damage is reported with its own reason, and any damage fails the check."""
    # Arrange: one good image, plus four broken cases.
    good_bytes = make_jpeg_bytes()
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "ok.jpg").write_bytes(good_bytes)
    (images_dir / "bad.jpg").write_bytes(b"not a jpeg")
    (images_dir / "changed.jpg").write_bytes(good_bytes)

    manifest = tmp_path / "manifest_val.jsonl"
    write_manifest(
        [
            make_record("images/ok.jpg", sha256_of(good_bytes)),          # passes
            make_record("images/missing.jpg", "0" * 64),                   # file absent
            make_record("images/bad.jpg", sha256_of(b"not a jpeg")),       # does not decode
            make_record("images/changed.jpg", "f" * 64),                   # hash differs
            make_record("images/ok.jpg"),                                  # hash never recorded
        ],
        manifest,
    )

    # Act
    report = verify_raw(manifest, tmp_path)

    # Assert
    assert report.checked == 5
    assert report.passed == 1
    assert report.ok is False

    reasons = [reason for _path, reason in report.failures]
    assert "missing" in reasons
    assert "sha256 mismatch" in reasons
    assert "no recorded sha256" in reasons
    assert any(reason.startswith("undecodable") for reason in reasons)


def test_verify_raw_reports_an_empty_manifest_as_a_failure(tmp_path):
    """An empty manifest is not evidence that anything was downloaded."""
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("")

    report = verify_raw(manifest, tmp_path)

    assert report.checked == 0
    assert report.ok is False


# ----------------------------------------------------------------------------------------------
# 5. Per-image licence
# ----------------------------------------------------------------------------------------------

def test_coco_records_keep_each_images_own_flickr_licence():
    """COCO images carry different Flickr licences (many NonCommercial or NoDerivs).

    The dataset-wide licence string is not enough: each record must name its own image's licence
    so a commercial export can exclude the images it may not use.
    """
    # Arrange: two images under different licences, each with one person box.
    annotations = {
        "licenses": [
            {"id": 1, "name": "Attribution-NonCommercial-ShareAlike License"},
            {"id": 4, "name": "Attribution License"},
        ],
        "categories": [{"id": 1, "name": "person"}],
        "images": [
            {"id": 10, "file_name": "a.jpg", "width": 100, "height": 100, "license": 1,
             "coco_url": "http://example.invalid/a.jpg", "flickr_url": "http://flickr.invalid/a"},
            {"id": 11, "file_name": "b.jpg", "width": 100, "height": 100, "license": 4,
             "coco_url": "http://example.invalid/b.jpg", "flickr_url": "http://flickr.invalid/b"},
        ],
        "annotations": [
            {"id": 100, "image_id": 10, "category_id": 1, "bbox": [0, 0, 50, 50], "iscrowd": 0},
            {"id": 101, "image_id": 11, "category_id": 1, "bbox": [0, 0, 50, 50], "iscrowd": 0},
        ],
    }

    # Act
    records = coco.select_records(annotations, LabelMap.load(), per_class_cap=10)

    # Assert
    licence_by_id = {record.image_id: record.extra["image_license"] for record in records}
    assert licence_by_id == {
        "coco:10": "Attribution-NonCommercial-ShareAlike License",
        "coco:11": "Attribution License",
    }
    flickr_by_id = {record.image_id: record.extra["flickr_url"] for record in records}
    assert flickr_by_id["coco:10"] == "http://flickr.invalid/a"


def test_coco_records_keep_source_annotation_ids():
    """Every converted box must trace back to its original COCO annotation id."""
    annotations = {
        "licenses": [{"id": 4, "name": "Attribution License"}],
        "categories": [{"id": 1, "name": "person"}],
        "images": [{"id": 10, "file_name": "a.jpg", "width": 100, "height": 100, "license": 4}],
        "annotations": [
            {"id": 555, "image_id": 10, "category_id": 1, "bbox": [0, 0, 50, 50], "iscrowd": 0},
            {"id": 556, "image_id": 10, "category_id": 1, "bbox": [10, 10, 40, 40], "iscrowd": 0},
        ],
    }

    records = coco.select_records(annotations, LabelMap.load(), per_class_cap=10)

    assert len(records) == 1
    assert records[0].extra["annotation_ids"] == "555,556"
