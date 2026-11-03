"""Regression tests for splits, cleaning boundaries, manifests and the access-state ladder."""

from __future__ import annotations

import json
from collections import Counter

import pytest
from ml.datasets.cleaning import clean
from ml.datasets.manifest import BoxRecord, ImageRecord, read_manifest, write_manifest
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
