"""Manifest schema (ticket 303): every sample must name its dataset version, licence, checksum, label-map version
and annotation source. A sample without a licence_id or a checksum must fail validation.
"""

from __future__ import annotations

import json

import pytest
from ml.datasets.acquisition import coco
from ml.datasets.label_map import LabelMap
from ml.datasets.manifest import (
    ANNOTATION_SOURCES,
    BoxRecord,
    ImageRecord,
    read_manifest,
    validate_records,
    write_manifest,
)


def complete_record(**changes) -> ImageRecord:
    """A record with every required field filled in; ``changes`` override single fields."""
    record = ImageRecord(
        image_id="coco:1",
        source="coco",
        path="images/1.jpg",
        width=100,
        height=100,
        group_id="coco:1",
        boxes=(BoxRecord("person", (0.0, 0.0, 10.0, 10.0), "person"),),
        sha256="a" * 64,
        dataset_version="2017",
        licence_id="CC-BY-2.0",
        label_map_version="1.0.0",
        annotation_source="source_dataset",
    )
    return record.with_(**changes)


def test_a_complete_record_passes():
    assert validate_records([complete_record()]) == []


@pytest.mark.parametrize("field", ["licence_id", "sha256", "dataset_version", "label_map_version",
                                   "annotation_source", "source", "path"])
def test_a_record_missing_a_required_field_fails(field):
    errors = validate_records([complete_record(**{field: ""})])
    assert errors == [f"missing_{field}:coco:1"]


def test_an_unknown_annotation_source_fails():
    errors = validate_records([complete_record(annotation_source="guessed")])
    assert errors == ["unknown_annotation_source:coco:1:guessed"]


def test_a_synthetic_record_must_say_its_annotations_are_synthetic():
    errors = validate_records([complete_record(is_synthetic=True)])
    assert errors == ["synthetic_flag_mismatch:coco:1"]
    assert validate_records([complete_record(is_synthetic=True, annotation_source="synthetic")]) == []


def test_split_is_required_only_when_asked():
    record = complete_record()
    assert validate_records([record]) == []
    assert validate_records([record], require_split=True) == ["missing_split:coco:1"]
    assert validate_records([record.with_(split="train")], require_split=True) == []


def test_annotation_sources_include_human_review_and_reject_nothing_silently():
    assert "human_reviewed" in ANNOTATION_SOURCES
    assert {"synthetic", "heuristic", "model_prediction"} <= set(ANNOTATION_SOURCES)


def test_manifests_written_before_this_schema_are_still_readable(tmp_path):
    """Version-1 manifests load, and then fail validation until they are regenerated."""
    path = tmp_path / "old.jsonl"
    old_line = {
        "manifest_version": "maxsight.manifest/1", "image_id": "coco:1", "source": "coco", "path": "images/1.jpg",
        "width": 100, "height": 100, "group_id": "coco:1", "boxes": [], "ignore_regions": [], "url": "",
        "sha256": "a" * 64, "dhash": "", "split": "", "is_synthetic": False, "license": "", "extra": {},
    }
    path.write_text(json.dumps(old_line) + "\n")

    records = list(read_manifest(path))

    assert len(records) == 1
    assert "missing_licence_id:coco:1" in validate_records(records)


def test_new_fields_round_trip(tmp_path):
    path = tmp_path / "m.jsonl"
    write_manifest([complete_record()], path)
    assert list(read_manifest(path)) == [complete_record()]


def test_coco_records_carry_the_schema_fields():
    annotations = {
        "licenses": [{"id": 3, "url": "http://creativecommons.org/licenses/by-nc-nd/2.0/",
                      "name": "Attribution-NonCommercial-NoDerivs License"}],
        "categories": [{"id": 1, "name": "person"}],
        "images": [{"id": 10, "file_name": "a.jpg", "width": 100, "height": 100, "license": 3}],
        "annotations": [{"id": 5, "image_id": 10, "category_id": 1, "bbox": [0, 0, 50, 50], "iscrowd": 0}],
    }
    label_map = LabelMap.load()

    record = coco.select_records(annotations, label_map, per_class_cap=10)[0]

    assert record.dataset_version == "2017"
    assert record.licence_id == "CC-BY-NC-ND-2.0"
    assert record.label_map_version == label_map.version != ""
    assert record.annotation_source == "source_dataset"


@pytest.mark.parametrize(("url", "expected"), [
    ("http://creativecommons.org/licenses/by-nc-sa/2.0/", "CC-BY-NC-SA-2.0"),
    ("http://creativecommons.org/licenses/by/2.0/", "CC-BY-2.0"),
    ("http://flickr.com/commons/usage/", "flickr-no-known-copyright-restrictions"),
    ("http://www.usa.gov/copyright.shtml", "us-government-work"),
    ("", ""),
])
def test_coco_licence_urls_become_licence_ids(url, expected):
    assert coco.licence_id_from_url(url) == expected


def test_openimages_records_pass_validation(tmp_path):
    import io

    from ml.datasets.acquisition import openimages_manual as oim
    from PIL import Image

    image = tmp_path / "images" / "validation" / "abc.jpg"
    image.parent.mkdir(parents=True)
    buffer = io.BytesIO()
    Image.new("RGB", (8, 6)).save(buffer, format="JPEG")
    image.write_bytes(buffer.getvalue())
    door = oim.BoxRow(label_id="/m/door", xmin=0.1, xmax=0.5, ymin=0.1, ymax=0.5,
                      is_group_of=False, is_depiction=False)
    selection = oim.Selection(images={"abc": [door]}, class_counts={})

    records = oim.to_records(selection, {"/m/door": "Door"}, "validation", tmp_path, LabelMap.load())

    assert validate_records(records) == []
    assert records[0].licence_id == "CC-BY-2.0" and records[0].dataset_version == "v7"


def test_bdd100k_records_fail_validation_until_images_are_on_disk(tmp_path):
    from ml.datasets.annotation import bdd100k
    from PIL import Image

    labels = [{"name": "abc-1.jpg", "attributes": {},
               "labels": [{"category": "car", "box2d": {"x1": 1, "y1": 1, "x2": 20, "y2": 20}}]}]
    labels_path = tmp_path / "det.json"
    labels_path.write_text(json.dumps(labels))

    without_images = bdd100k.to_records(labels_path, LabelMap.load())
    assert validate_records(without_images) == ["missing_sha256:bdd100k:abc-1.jpg"]

    image = tmp_path / "images" / "100k" / "val" / "abc-1.jpg"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (32, 24)).save(image)
    with_images = bdd100k.to_records(labels_path, LabelMap.load(), image_root=tmp_path)
    assert validate_records(with_images) == []
