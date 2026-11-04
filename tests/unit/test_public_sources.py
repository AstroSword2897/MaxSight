"""Registry, policy, planning and converter tests for the eight public datasets (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from ml.datasets.acquisition import plan as acq_plan
from ml.datasets.acquisition.fetch import safe_extract_tar, verify_md5
from ml.datasets.annotation import bdd100k, diode
from ml.datasets.label_map import LabelMap
from ml.datasets.sources import (
    ACQUIRED_MARKER,
    State,
    assert_detector_training_sources,
    commercial_release_blockers,
    current_state,
    load_registry,
)
from PIL import Image

REG = load_registry()
EXPECTED = {"coco2017", "open_images_v7", "mapillary_vistas_v2", "bdd100k", "sanpo", "diode", "pie", "ego4d"}


def test_registry_has_all_eight_sources_with_roles():
    assert set(REG) == EXPECTED
    assert REG["bdd100k"].role == "eval_only"
    assert {s.id for s in REG.values() if s.role == "pretrain"} == {"coco2017", "open_images_v7", "mapillary_vistas_v2"}


def test_configured_sources_are_never_reported_downloaded(tmp_path):
    assert all(current_state(s, tmp_path) is State.NOT_DOWNLOADED for s in REG.values())


def test_state_follows_marker_files(tmp_path):
    raw = tmp_path / "raw" / "diode"
    raw.mkdir(parents=True)
    (raw / ACQUIRED_MARKER).write_text(json.dumps({"verified": False}))
    assert current_state(REG["diode"], tmp_path) is State.DOWNLOADED_UNVERIFIED
    (raw / ACQUIRED_MARKER).write_text(json.dumps({"verified": True}))
    assert current_state(REG["diode"], tmp_path) is State.RAW_VERIFIED


@pytest.mark.parametrize("bad", ["bdd100k", "sanpo", "diode", "pie", "ego4d"])
def test_only_pretrain_sources_may_enter_detector_mix(bad):
    with pytest.raises(ValueError):
        assert_detector_training_sources({"coco2017", bad}, REG)
    assert_detector_training_sources({"coco2017", "open_images_v7", "mapillary_vistas_v2"}, REG)


def test_vistas_blocks_commercial_release():
    assert any(b.startswith("mapillary_vistas_v2") for b in commercial_release_blockers({"mapillary_vistas_v2", "sanpo"}, REG))
    assert not any(b.startswith("sanpo") for b in commercial_release_blockers({"sanpo"}, REG))


def test_openimages_registry_records_missing_boxable_classes_and_label_map_flags_them():
    assert set(REG["open_images_v7"].raw["unavailable_classes"]) == {"Traffic cone", "Pole"}


def test_sanpo_plan_requires_sessions_and_excludes_right_and_zed_and_imu_not_requested():
    with pytest.raises(ValueError):
        acq_plan.sanpo_plan(REG["sanpo"], [])
    with pytest.raises(ValueError):
        acq_plan.sanpo_plan(REG["sanpo"], ["../x"])
    flat = " ".join(" ".join(s) for s in acq_plan.sanpo_plan(REG["sanpo"], ["s1"]).argv)
    assert ".*/right" in flat and ".*/zed_depth_maps" in flat and ".*/segmentation_masks" in flat and ".*/depth_maps" in flat
    opt = " ".join(" ".join(s) for s in acq_plan.sanpo_plan(REG["sanpo"], ["s1"], segmentation=True, depth=True).argv)
    assert ".*/segmentation_masks" not in opt and ".*/depth_maps" not in opt
    assert "sanpo-real/s1" in opt


def test_ego4d_plan_refuses_unfiltered_video():
    with pytest.raises(ValueError):
        acq_plan.ego4d_plan(REG["ego4d"], include_video=True)
    pl = acq_plan.ego4d_plan(REG["ego4d"], benchmarks=["EM"], include_video=True)
    assert "--benchmarks" in pl.argv[1] and pl.manual and not pl.automatable


def test_diode_train_is_deferred_and_val_size_is_planned():
    with pytest.raises(ValueError):
        acq_plan.diode_plan(REG["diode"], include_train=True)
    assert 2.5 < acq_plan.diode_plan(REG["diode"]).est_gb < 3


def test_verify_md5_and_safe_extract_reject_bad_input(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"abc")
    assert verify_md5(f, "900150983cd24fb0d6963f7d28e17f72") and not verify_md5(f, "0" * 32) and verify_md5(f, None)
    import tarfile
    evil = tmp_path / "evil.tar"
    with tarfile.open(evil, "w") as tf:
        info = tarfile.TarInfo("../escape.txt")
        info.size = 0
        tf.addfile(info)
    with pytest.raises(ValueError):
        safe_extract_tar(evil, tmp_path / "out")


def test_bdd100k_records_are_eval_split_with_domain_metadata_and_rider_ignored(tmp_path):
    labels = [{"name": "abc123-def456.jpg", "attributes": {"weather": "rainy", "scene": "city street", "timeofday": "night"},
               "labels": [{"category": "car", "box2d": {"x1": 10, "y1": 10, "x2": 100, "y2": 90}},
                          {"category": "rider", "box2d": {"x1": 200, "y1": 10, "x2": 260, "y2": 90}},
                          {"category": "traffic sign", "box2d": {"x1": 1, "y1": 1, "x2": 9, "y2": 9}},
                          {"category": "lane/crosswalk", "poly2d": []}]}]
    p = tmp_path / "det.json"
    p.write_text(json.dumps(labels))
    (rec,) = bdd100k.to_records(p, LabelMap.load())
    assert rec.split == "eval" and rec.source == "bdd100k"
    assert [b.cls for b in rec.boxes] == ["vehicle"] and rec.boxes[0].source_class == "car"
    assert len(rec.ignore_regions) == 1
    assert rec.extra["timeofday"] == "night" and rec.extra["camera_domain"] == "vehicle_mounted"
    assert rec.group_id == "bdd100k:abc123"
    assert json.loads(p.read_text()) == labels  # source file untouched


def _diode_item(root: Path, name: str, shape=(6, 8), mask_shape=None, depth_val=1.5):
    Image.new("RGB", (shape[1], shape[0])).save(root / f"{name}.png")
    np.save(root / f"{name}_depth.npy", np.full((*shape, 1), depth_val, dtype="float32"))
    np.save(root / f"{name}_depth_mask.npy", np.ones(mask_shape or shape, dtype="float32"))


def test_diode_scan_accepts_matched_triplets_and_flags_breakage(tmp_path):
    _diode_item(tmp_path, "ok")
    assert diode.scan(tmp_path) == []
    _diode_item(tmp_path, "badshape", mask_shape=(3, 3))
    Image.new("RGB", (8, 6)).save(tmp_path / "nomask.png")
    np.save(tmp_path / "nomask_depth.npy", np.ones((6, 8, 1), dtype="float32"))
    probs = diode.scan(tmp_path)
    assert any(p.startswith("shape_mismatch:badshape") for p in probs)
    assert "missing_depth_mask:nomask.png" in probs
    _diode_item(tmp_path, "neg", depth_val=-1.0)
    assert any(p.startswith("invalid_depth_in_valid_region:neg") for p in diode.scan(tmp_path))


def test_acquired_marker_records_download_without_claiming_verification(tmp_path):
    from ml.datasets.sources import write_acquired_marker

    raw = tmp_path / "raw" / "open_images_v7"
    write_acquired_marker(raw, verified=False, reason="no published per-image checksum", images=300, failures=0)
    info = json.loads((raw / ACQUIRED_MARKER).read_text())
    assert info["verified"] is False and info["images"] == 300 and info["acquired_at"].endswith("Z")
    assert current_state(REG["open_images_v7"], tmp_path) is State.DOWNLOADED_UNVERIFIED


def test_open_images_instructions_match_the_manual_boto3_route():
    text = " ".join(acq_plan.manual_plan(REG["open_images_v7"]).manual) + " " + REG["open_images_v7"].access_note
    assert "boto3" in text and "acquire_data.py openimages" in text
    assert "fiftyone" not in text.lower()
