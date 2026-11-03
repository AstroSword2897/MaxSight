"""Cross-file consistency of the data configuration (label map, source registry, training registry)."""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO = Path(__file__).resolve().parents[2]


def _load(rel: str) -> dict:
    return yaml.safe_load((REPO / rel).read_text(encoding="utf-8"))


LABEL_MAP = _load("configs/data/label_map.yaml")
SOURCES = _load("configs/data/sources.yaml")["sources"]
# Names checked against the official oidv7-class-descriptions-boxable.csv on 2026-11-02.
OPEN_IMAGES_BOXABLE = {
    "Person", "Bicycle", "Motorcycle", "Car", "Bus", "Truck", "Dog", "Cat", "Chair", "Couch", "Table",
    "Bench", "Houseplant", "Box", "Suitcase", "Backpack", "Handbag", "Waste container", "Street light",
    "Fire hydrant", "Door", "Stairs",
}


def test_detector_taxonomy_is_the_agreed_seventeen_classes():
    assert LABEL_MAP["classes"] == [
        "person", "bicycle", "motorcycle", "vehicle", "animal", "chair", "couch", "table", "bench",
        "potted_plant", "bag_box", "trash_bin", "pole_bollard", "fire_hydrant", "traffic_cone", "door", "stairs",
    ]


@pytest.mark.parametrize("source", sorted(LABEL_MAP["sources"]))
def test_every_mapping_target_is_in_the_class_space(source):
    assert set(LABEL_MAP["sources"][source].values()) <= set(LABEL_MAP["classes"])


def test_label_map_sources_are_registered_sources():
    registered_labels = {s.get("label_source") for s in SOURCES.values()}
    assert set(LABEL_MAP["sources"]) <= registered_labels


def test_ignore_lists_never_overlap_mapped_names():
    for src, ignored in LABEL_MAP.get("ignore", {}).items():
        assert not set(ignored) & set(LABEL_MAP["sources"].get(src, {})), src


def test_open_images_names_in_label_map_are_boxable_or_declared_unavailable():
    unavailable = set(SOURCES["open_images_v7"]["unavailable_classes"])
    unknown = set(LABEL_MAP["sources"]["openimages"]) - OPEN_IMAGES_BOXABLE - unavailable
    assert not unknown, unknown


def test_unavailable_open_images_classes_were_actually_requested():
    oi = SOURCES["open_images_v7"]
    assert set(oi["unavailable_classes"]) <= set(oi["requested_classes"])


@pytest.mark.xfail(strict=True, reason="DR-202 gap: traffic_cone has no usable source (Open Images has no Traffic cone)")
def test_every_detector_class_has_at_least_one_usable_source():
    covered = set()
    for src, mapping in LABEL_MAP["sources"].items():
        usable = {n: c for n, c in mapping.items()
                  if not (src == "openimages" and n not in OPEN_IMAGES_BOXABLE)}
        covered |= set(usable.values())
    assert set(LABEL_MAP["classes"]) <= covered


def test_dr112_dataset_roles():
    roles = {k: v["role"] for k, v in SOURCES.items()}
    assert {k for k, r in roles.items() if r == "pretrain"} == {"coco2017", "open_images_v7", "mapillary_vistas_v2"}
    assert roles["bdd100k"] == "eval_only"
    assert {k for k, r in roles.items() if r == "research"} == {"sanpo", "diode", "pie", "ego4d"}


def test_vistas_is_never_marked_commercially_usable():
    assert SOURCES["mapillary_vistas_v2"]["license"]["commercial_use"] is False


@pytest.mark.parametrize("sid", sorted(SOURCES))
def test_every_source_records_license_access_and_homepage(sid):
    s = SOURCES[sid]
    assert s["homepage"].startswith("https://")
    assert s["license"]["id"] and "commercial_use" in s["license"]
    assert s["access"]["kind"] in {"automatic", "registration", "license_acceptance", "credentials", "manual_download"}


def test_open_images_per_class_cap_matches_plan():
    assert SOURCES["open_images_v7"]["max_images_per_class"] == 4000


def test_diode_validation_archive_pins_published_md5():
    val = SOURCES["diode"]["archives"]["val"]
    assert len(val["md5"]) == 32 and val["url"].endswith("val.tar.gz")
    assert "train" in SOURCES["diode"]["deferred_archives"]


def test_training_registry_num_classes_match_label_spaces():
    spaces = {s["id"]: s["num_classes"] for s in _load("ml/training/configs/registry/label_spaces.yaml")["label_spaces"]}
    for ds in _load("ml/training/configs/registry/datasets.yaml")["datasets"]:
        if ds.get("label_space"):
            assert ds["num_classes"] == spaces[ds["label_space"]], ds["id"]
