"""Regression tests for splits, cleaning boundaries, manifests and the access-state ladder."""

from __future__ import annotations

import json
from collections import Counter

import pytest
from ml.datasets.cleaning import clean
from ml.datasets.manifest import BoxRecord, ImageRecord, read_manifest, write_manifest
from ml.datasets.sources import (
    ACQUIRED_MARKER,
    VALIDATED_MARKER,
    State,
    current_state,
    load_registry,
)
from ml.datasets.splits import assign_splits
from PIL import Image


def _rec(i: int, **kw) -> ImageRecord:
    return ImageRecord(f"x:{i}", "x", f"{i}.png", 100, 100, kw.pop("group_id", f"g:{i}"),
                       kw.pop("boxes", (BoxRecord("person", (0.0, 0.0, 50.0, 50.0)),)), **kw)


def test_split_ratios_are_close_to_requested_on_many_groups():
    out = assign_splits([_rec(i) for i in range(4000)], ratios=(0.7, 0.15, 0.15))
    share = {k: v / 4000 for k, v in Counter(r.split for r in out).items()}
    assert abs(share["train"] - 0.70) < 0.04 and abs(share["val"] - 0.15) < 0.03 and abs(share["test"] - 0.15) < 0.03


def test_split_changes_with_seed_but_not_between_runs():
    recs = [_rec(i) for i in range(300)]
    a, b = assign_splits(recs, seed=1), assign_splits(recs, seed=1)
    c = assign_splits(recs, seed=2)
    assert [r.split for r in a] == [r.split for r in b]
    assert [r.split for r in a] != [r.split for r in c]


def test_split_rejects_ratios_that_do_not_sum_to_one():
    with pytest.raises(ValueError):
        assign_splits([_rec(1)], ratios=(0.5, 0.5, 0.5))


def test_frames_of_one_video_never_straddle_splits():
    recs = [_rec(i, group_id=f"video:{i % 25}") for i in range(500)]
    seen: dict[str, set[str]] = {}
    for r in assign_splits(recs):
        seen.setdefault(r.group_id, set()).add(r.split)
    assert all(len(s) == 1 for s in seen.values())


def test_holdout_environment_is_isolated_from_training():
    recs = [_rec(i, extra={"environment": "hospital" if i % 5 == 0 else "street"}) for i in range(100)]
    out = assign_splits(recs, holdout_groups=frozenset({"hospital"}))
    assert {r.split for r in out if r.extra["environment"] == "hospital"} == {"test_holdout"}
    assert "test_holdout" not in {r.split for r in out if r.extra["environment"] == "street"}


@pytest.mark.parametrize(("size", "kept"), [(8.0, True), (7.9, False)])
def test_clean_min_box_boundary(tmp_path, size, kept):
    rec = _rec(1, boxes=(BoxRecord("person", (0.0, 0.0, size, 50.0)),))
    out, rep = clean([rec], tmp_path, verify_files=False)
    assert bool(out) is kept and (rep.boxes_dropped_small == 0) is kept


def test_clean_drops_degenerate_boxes_and_reports_them(tmp_path):
    rec = _rec(1, boxes=(BoxRecord("person", (10.0, 10.0, 10.0, 60.0)),))
    out, rep = clean([rec], tmp_path, verify_files=False)
    assert out == [] and rep.dropped["no_boxes_after_filter"] == 1


def test_clean_corrects_declared_size_from_the_actual_file(tmp_path):
    Image.new("RGB", (64, 32)).save(tmp_path / "1.png")
    (out,), _ = clean([_rec(1, boxes=(BoxRecord("person", (0.0, 0.0, 20.0, 20.0)),))], tmp_path)
    assert (out.width, out.height) == (64, 32) and len(out.sha256) == 64


def test_manifest_round_trip_preserves_every_field(tmp_path):
    rec = _rec(1, ignore_regions=((1.0, 2.0, 3.0, 4.0),), url="http://u", is_synthetic=True,
               license="L", extra={"weather": "rainy"})
    path = tmp_path / "m.jsonl"
    assert write_manifest([rec], path) == 1
    assert list(read_manifest(path)) == [rec]


def test_manifest_rejects_unknown_version(tmp_path):
    path = tmp_path / "m.jsonl"
    path.write_text(json.dumps({"manifest_version": "other/9"}) + "\n")
    with pytest.raises(ValueError, match="unsupported manifest version"):
        list(read_manifest(path))


def test_assign_splits_preserves_eval_only_records():
    """BDD100K records are tagged split='eval'; a hash split must not move them into train."""
    recs = [_rec(i, split="eval") for i in range(200)] + [_rec(1000 + i) for i in range(200)]
    out = assign_splits(recs)
    assert {r.split for r in out[:200]} == {"eval"}
    assert {r.split for r in out[200:]} <= {"train", "val", "test"}


def test_clean_turns_small_boxes_into_ignore_regions_not_false_background(tmp_path):
    """Found on real Open Images data: 8 small boxes vanished, leaving real objects labelled as background."""
    big = BoxRecord("person", (0.0, 0.0, 50.0, 50.0))
    tiny = BoxRecord("person", (60.0, 60.0, 64.0, 64.0))
    (out,), rep = clean([_rec(1, boxes=(big, tiny))], tmp_path, verify_files=False)
    assert out.boxes == (big,)
    assert (60.0, 60.0, 64.0, 64.0) in out.ignore_regions
    assert rep.boxes_dropped_small == 1


def test_audit_finds_cross_split_near_duplicates_at_realistic_size():
    """Above 3000 images the audit used to skip this check and still pass (seen on the real Open Images run)."""
    import random

    from ml.datasets.audit import audit
    from ml.datasets.label_map import LabelMap

    rng = random.Random(0)
    classes = LabelMap.load().classes
    recs = [_rec(i, split="train" if i % 2 else "test", dhash=f"{rng.getrandbits(64):016x}",
                 boxes=(BoxRecord(classes[i % len(classes)], (0.0, 0.0, 50.0, 50.0)),)) for i in range(4000)]
    twin = int(recs[0].dhash, 16) ^ 1  # one bit away from a test image, placed in train
    recs.append(_rec(9999, split="train", dhash=f"{twin:016x}"))
    res = audit(recs, LabelMap.load(), max_class_share=1.0)
    assert any(e.startswith("near_duplicate_across_splits") for e in res.errors)
    assert not any("skipped" in w for w in res.warnings)


def test_clean_rejects_a_dup_threshold_its_index_cannot_honour(tmp_path):
    with pytest.raises(ValueError, match="dup_hamming"):
        clean([_rec(1)], tmp_path, dup_hamming=8, verify_files=False)


def test_coco_byte_budget_reports_the_images_it_did_not_download(tmp_path, monkeypatch):
    from ml.datasets.acquisition import coco

    monkeypatch.setattr(coco, "_download_one", lambda rec, root, retries=3: (rec, 2 * 1024**3, ""))
    ok, failed = coco.download_images([_rec(i) for i in range(100)], tmp_path, max_gb=1, workers=1)
    assert len(ok) + len(failed) == 100
    assert failed and all(reason == "byte budget reached" for _, reason in failed)


def test_access_state_ladder_reaches_validated_only_with_evidence(tmp_path):
    src = load_registry()["bdd100k"]
    assert current_state(src, tmp_path) is State.NOT_DOWNLOADED
    raw = tmp_path / "raw" / src.id
    raw.mkdir(parents=True)
    (raw / ACQUIRED_MARKER).write_text(json.dumps({"verified": False}))
    assert current_state(src, tmp_path) is State.DOWNLOADED_UNVERIFIED
    (raw / ACQUIRED_MARKER).write_text(json.dumps({"verified": True}))
    assert current_state(src, tmp_path) is State.RAW_VERIFIED
    processed = tmp_path / "processed" / src.id
    processed.mkdir(parents=True)
    assert current_state(src, tmp_path) is State.RAW_VERIFIED  # empty processed dir is not evidence
    (processed / "manifest.jsonl").write_text("")
    assert current_state(src, tmp_path) is State.PREPROCESSED_UNVERIFIED
    (processed / VALIDATED_MARKER).write_text("{}")
    assert current_state(src, tmp_path) is State.VALIDATED
