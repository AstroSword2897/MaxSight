"""Unit tests for dataset acquisition selection, cleaning, splits and audit (no network)."""

from __future__ import annotations

import random
from pathlib import Path

import pytest
from ml.datasets.acquisition.coco import select_records
from ml.datasets.audit import audit
from ml.datasets.cleaning import clean, dhash, hamming
from ml.datasets.label_map import LabelMap
from ml.datasets.manifest import BoxRecord, ImageRecord, read_manifest, write_manifest
from ml.datasets.splits import assign_splits
from PIL import Image

LM = LabelMap.load()


def _coco(n_person_only: int, n_chair: int, n_hydrant: int) -> dict:
    cats = [{"id": 1, "name": "person"}, {"id": 62, "name": "chair"}, {"id": 10, "name": "fire hydrant"},
            {"id": 65, "name": "bed"}]
    images, anns = [], []

    def add(i, cid, extra=()):
        images.append({"id": i, "file_name": f"{i:012d}.jpg", "width": 640, "height": 480, "coco_url": f"http://x/{i}.jpg"})
        anns.append({"image_id": i, "category_id": cid, "bbox": [10, 10, 100, 100], "iscrowd": 0})
        for c in extra:
            anns.append({"image_id": i, "category_id": c, "bbox": [200, 200, 50, 50], "iscrowd": 0})

    i = 0
    for _ in range(n_person_only):
        i += 1
        add(i, 1)
    for _ in range(n_chair):
        i += 1
        add(i, 62, extra=(1,))
    for _ in range(n_hydrant):
        i += 1
        add(i, 10)
    i += 1
    add(i, 65)  # bed only -> unmapped, no boxes
    return {"categories": cats, "images": images, "annotations": anns}


def test_label_map_rejects_classes_outside_class_space(tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text("classes: [a]\nsources:\n  s: {x: b}\n")
    with pytest.raises(ValueError):
        LabelMap.load(p)


def test_select_keeps_rare_classes_when_common_class_is_capped():
    recs = select_records(_coco(200, 30, 10), LM, per_class_cap=20, seed=1)
    classes = [b.cls for r in recs for b in r.boxes]
    assert classes.count("fire_hydrant") == 10  # every rare instance kept
    assert classes.count("chair") >= 20
    assert not any(r.image_id.endswith(":241") for r in recs)  # bed-only image not selected


def test_select_keeps_all_boxes_of_an_accepted_image():
    recs = select_records(_coco(0, 5, 0), LM, per_class_cap=2, seed=0)
    for r in recs:
        assert {b.cls for b in r.boxes} == {"chair", "person"}  # no false background


def _img(path: Path, seed: int, size=(64, 64)) -> None:
    rng = random.Random(seed)
    im = Image.new("RGB", size)
    im.putdata([(rng.randrange(256),) * 3 for _ in range(size[0] * size[1])])
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def _rec(i, path, box=(5, 5, 40, 40), group=None, cls="chair"):
    return ImageRecord(f"t:{i}", "t", path, 64, 64, group or f"g{i}", (BoxRecord(cls, tuple(map(float, box))),))


def test_clean_drops_corrupt_duplicate_small_and_missing(tmp_path):
    _img(tmp_path / "a.png", 1)
    (tmp_path / "bad.png").write_bytes(b"not an image")
    Image.open(tmp_path / "a.png").save(tmp_path / "a_copy.png")
    _img(tmp_path / "b.png", 2)
    recs = [_rec(1, "a.png"), _rec(2, "bad.png"), _rec(3, "a_copy.png"), _rec(4, "b.png", box=(1, 1, 4, 4)),
            _rec(5, "missing.png"), _rec(6, "b.png")]
    kept, rep = clean(recs, tmp_path, min_box_px=8)
    assert [r.image_id for r in kept] == ["t:1", "t:6"]
    assert rep.dropped["corrupt_or_missing"] == 2
    assert rep.dropped["near_duplicate"] == 1
    assert rep.dropped["no_boxes_after_filter"] == 1
    assert all(r.sha256 and r.dhash for r in kept)


def test_dhash_detects_resized_copy_as_near_duplicate(tmp_path):
    _img(tmp_path / "a.png", 7, size=(128, 128))
    a = Image.open(tmp_path / "a.png")
    assert hamming(dhash(a), dhash(a.resize((64, 64)))) <= 8


def test_cap_per_class_drops_whole_images(tmp_path):
    recs = []
    for i in range(10):
        _img(tmp_path / f"{i}.png", 100 + i)
        recs.append(_rec(i, f"{i}.png"))
    kept, rep = clean(recs, tmp_path, per_class_cap=4)
    assert len(kept) == 4 and rep.dropped["class_cap"] == 6


def test_splits_never_separate_a_group_and_are_deterministic():
    recs = [_rec(i, "x", group=f"clip{i % 20}") for i in range(200)]
    a = assign_splits(recs, seed=3)
    b = assign_splits(recs, seed=3)
    assert [r.split for r in a] == [r.split for r in b]
    by_group: dict[str, set[str]] = {}
    for r in a:
        by_group.setdefault(r.group_id, set()).add(r.split)
    assert all(len(s) == 1 for s in by_group.values())
    assert {r.split for r in a} <= {"train", "val", "test"}


def test_holdout_environment_goes_to_test_holdout():
    recs = [_rec(i, "x", group=f"g{i}").with_(extra={"environment": "lobby" if i % 2 else "hall"}) for i in range(20)]
    out = assign_splits(recs, holdout_groups=frozenset({"lobby"}))
    assert all(r.split == "test_holdout" for r in out if r.extra["environment"] == "lobby")
    assert all(r.split != "test_holdout" for r in out if r.extra["environment"] == "hall")


def test_audit_flags_group_leak_and_class_skew():
    recs = [_rec(i, "x", group="same", cls="person").with_(split=s) for i, s in enumerate(["train", "val"])]
    recs += [_rec(10 + i, "x", cls="chair").with_(split="train") for i in range(2)]
    res = audit(recs, LM, max_class_share=0.6)
    assert any(e.startswith("group_leak:same") for e in res.errors)
    res2 = audit([_rec(i, "x", cls="person").with_(split="train") for i in range(5)], LM)
    assert any(e.startswith("class_share_exceeds") for e in res2.errors)


def test_audit_flags_unknown_class_and_out_of_bounds():
    r = ImageRecord("t:1", "t", "x", 64, 64, "g", (BoxRecord("spaceship", (0, 0, 10, 10)), BoxRecord("chair", (0, 0, 500, 10))))
    res = audit([r], LM)
    assert any(e.startswith("unknown_class") for e in res.errors)
    assert any(e.startswith("box_out_of_bounds") for e in res.errors)


def test_manifest_roundtrip(tmp_path):
    recs = [_rec(1, "a.png").with_(dhash="00ff", split="train", extra={"environment": "hall"})]
    write_manifest(recs, tmp_path / "m.jsonl")
    assert list(read_manifest(tmp_path / "m.jsonl")) == recs


def test_small_mapped_boxes_become_ignore_regions_not_false_background():
    """A real but tiny person must not be silently dropped: unlabelled it would train as background."""
    ann = {"categories": [{"id": 1, "name": "person"}, {"id": 62, "name": "chair"}],
           "images": [{"id": 1, "file_name": "1.jpg", "width": 640, "height": 480, "coco_url": "http://x/1.jpg"}],
           "annotations": [{"image_id": 1, "category_id": 62, "bbox": [10, 10, 100, 100], "iscrowd": 0},
                           {"image_id": 1, "category_id": 1, "bbox": [300, 300, 4, 4], "iscrowd": 0}]}
    (rec,) = select_records(ann, LM, per_class_cap=10)
    assert [b.cls for b in rec.boxes] == ["chair"]
    assert (300.0, 300.0, 304.0, 304.0) in rec.ignore_regions
