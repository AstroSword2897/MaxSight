"""Open Images manual route: class resolution, capped selection, list format, retrying download, records."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from ml.datasets.acquisition import openimages_manual as oim
from ml.datasets.label_map import LabelMap
from ml.datasets.sources import load_registry
from PIL import Image

CLASSES = "LabelName,DisplayName\n/m/door,Door\n/m/stairs,Stairs\n/m/box,Box\n/m/cat,Cat\n"
HEADER = ["ImageID", "Source", "LabelName", "Confidence", "XMin", "XMax", "YMin", "YMax",
          "IsOccluded", "IsTruncated", "IsGroupOf", "IsDepiction", "IsInside"]


def _bbox(path: Path, rows: list[tuple]) -> Path:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        for image_id, label, group, depiction in rows:
            w.writerow([image_id, "xclick", label, 1, 0.1, 0.5, 0.2, 0.6, 0, 0, int(group), int(depiction), 0])
    return path


def test_registry_urls_match_the_module_constants():
    urls = load_registry()["open_images_v7"].raw["urls"]
    assert urls == oim.URLS
    assert load_registry()["open_images_v7"].raw["s3_bucket"] == oim.BUCKET


def test_class_descriptions_parse_with_or_without_a_header_row():
    headerless = "/m/door,Door\n/m/stairs,Stairs\n"
    assert oim.parse_class_descriptions(CLASSES)["Door"] == "/m/door"
    assert oim.parse_class_descriptions(headerless) == {"Door": "/m/door", "Stairs": "/m/stairs"}
    assert "LabelName" not in oim.parse_class_descriptions(CLASSES).values()


def test_resolve_classes_flags_non_boxable_names_case_insensitively():
    found, missing = oim.resolve_classes(oim.parse_class_descriptions(CLASSES), ["door", "Traffic cone", "Pole", "Box"])
    assert found == {"/m/door": "Door", "/m/box": "Box"}
    assert missing == ["Traffic cone", "Pole"]


def test_selection_respects_the_cap_and_keeps_every_wanted_box_of_an_image(tmp_path):
    rows = [(f"img{i}", "/m/door", False, False) for i in range(10)] + [("img0", "/m/box", False, False)]
    wanted = {"/m/door": "Door", "/m/box": "Box"}
    sel = oim.select_images(oim.iter_bbox_rows(_bbox(tmp_path / "b.csv", rows), wanted), per_class_cap=3)
    assert sel.class_counts["/m/door"] <= 3 + 1  # an image is whole or not at all
    assert any(len(b) == 2 for b in sel.images.values()) or "img0" not in sel.images
    assert "/m/box" in sel.class_counts  # the rare class is visited first


def test_selection_is_deterministic_per_seed(tmp_path):
    rows = [(f"i{n}", "/m/door", False, False) for n in range(50)]
    p = _bbox(tmp_path / "b.csv", rows)
    a = oim.select_images(oim.iter_bbox_rows(p, {"/m/door": "Door"}), 5, seed=1)
    b = oim.select_images(oim.iter_bbox_rows(p, {"/m/door": "Door"}), 5, seed=1)
    c = oim.select_images(oim.iter_bbox_rows(p, {"/m/door": "Door"}), 5, seed=2)
    assert list(a.images) == list(b.images) and set(a.images) != set(c.images)


def test_max_images_stops_selection_and_class_counts_match_the_selected_images(tmp_path):
    rows = [(f"i{n:03d}", "/m/door", False, False) for n in range(100)]
    p = _bbox(tmp_path / "b.csv", rows)
    wanted = {"/m/door": "Door"}
    a = oim.select_images(oim.iter_bbox_rows(p, wanted), 1000, seed=1, max_images=10)
    b = oim.select_images(oim.iter_bbox_rows(p, wanted), 1000, seed=2, max_images=10)
    assert len(a.images) == 10 and a.class_counts["/m/door"] == sum(len(v) for v in a.images.values())
    assert set(a.images) != set(sorted(a.images)[:0]) and set(a.images) != set(b.images)  # not id-ordered, seed-driven
    assert set(a.images) != {f"i{n:03d}" for n in range(10)}


def test_unwanted_labels_are_never_loaded(tmp_path):
    p = _bbox(tmp_path / "b.csv", [("a", "/m/cat", False, False), ("b", "/m/door", False, False)])
    assert [i for i, _ in oim.iter_bbox_rows(p, {"/m/door": "Door"})] == ["b"]


def test_image_list_round_trips_in_the_official_format(tmp_path):
    p = tmp_path / "list.txt"
    assert oim.write_image_list("validation", ["bb", "aa"], p) == 2
    assert p.read_text() == "validation/aa\nvalidation/bb\n"
    assert oim.read_image_list(p) == [("validation", "aa"), ("validation", "bb")]


@pytest.mark.parametrize("bad", ["val/abc", "train/xyz!", "train", "train/abc/def"])
def test_image_list_rejects_malformed_lines_with_the_line_number(tmp_path, bad):
    p = tmp_path / "list.txt"
    p.write_text(f"train/abc123\n{bad}\n")
    with pytest.raises(ValueError, match="line 2"):
        oim.read_image_list(p)


def test_image_list_tolerates_jpg_suffix_and_blank_lines(tmp_path):
    p = tmp_path / "list.txt"
    p.write_text("\ntrain/abc123.jpg\n\n")
    assert oim.read_image_list(p) == [("train", "abc123")]


class _FakeS3:
    def __init__(self, fail_first: dict[str, int] | None = None, always_fail: set[str] | None = None):
        self.calls: list[str] = []
        self._fail_first = dict(fail_first or {})
        self._always = always_fail or set()

    def download_file(self, bucket, key, filename):
        self.calls.append(key)
        assert bucket == oim.BUCKET
        if key in self._always:
            raise RuntimeError("403")
        if self._fail_first.get(key, 0) > 0:
            self._fail_first[key] -= 1
            raise RuntimeError("timeout")
        Image.new("RGB", (40, 20)).save(filename, format="JPEG")


def test_download_keeps_the_split_in_the_path_and_never_redownloads(tmp_path):
    client = _FakeS3()
    pairs = [("validation", "aa"), ("train", "aa")]  # same id in two splits must not collide
    ok, failed = oim.download_images(pairs, tmp_path, client=client, workers=1)
    assert len(ok) == 2 and failed == []
    assert (tmp_path / "validation" / "aa.jpg").exists() and (tmp_path / "train" / "aa.jpg").exists()
    oim.download_images(pairs, tmp_path, client=client, workers=1)
    assert len(client.calls) == 2  # second run skipped finished files


def test_download_retries_transient_errors_and_collects_permanent_failures(tmp_path):
    client = _FakeS3(fail_first={"validation/aa.jpg": 2}, always_fail={"validation/bb.jpg"})
    ok, failed = oim.download_images([("validation", "aa"), ("validation", "bb")], tmp_path, client=client,
                                     workers=1, retries=3, sleep=lambda _s: None)
    assert ok == [("validation", "aa")]
    assert [k for k, _ in failed] == ["validation/bb"] and "403" in failed[0][1]
    assert not list(tmp_path.rglob("*.part"))


def test_download_never_overwrites_an_existing_file(tmp_path):
    (tmp_path / "validation").mkdir()
    (tmp_path / "validation" / "aa.jpg").write_bytes(b"original")
    client = _FakeS3()
    oim.download_images([("validation", "aa")], tmp_path, client=client, workers=1)
    assert client.calls == [] and (tmp_path / "validation" / "aa.jpg").read_bytes() == b"original"


def test_records_use_pixels_and_turn_group_and_depiction_boxes_into_ignore_regions(tmp_path):
    rows = [("img1", "/m/door", False, False), ("img1", "/m/stairs", True, False), ("img1", "/m/box", False, True)]
    wanted = {"/m/door": "Door", "/m/stairs": "Stairs", "/m/box": "Box"}
    sel = oim.select_images(oim.iter_bbox_rows(_bbox(tmp_path / "b.csv", rows), wanted), 10)
    (tmp_path / "images" / "validation").mkdir(parents=True)
    Image.new("RGB", (200, 100)).save(tmp_path / "images" / "validation" / "img1.jpg")
    (rec,) = oim.to_records(sel, wanted, "validation", tmp_path, LabelMap.load())
    assert [b.cls for b in rec.boxes] == ["door"] and rec.boxes[0].source_class == "Door"
    assert rec.boxes[0].bbox_xyxy == pytest.approx((20.0, 20.0, 100.0, 60.0))
    assert len(rec.ignore_regions) == 2 and rec.path == "images/validation/img1.jpg"
    assert rec.license.startswith("annotations CC BY 4.0") and rec.extra == {"split": "validation"}


def test_images_missing_on_disk_produce_no_record(tmp_path):
    sel = oim.Selection(images={"ghost": [oim.BoxRow("/m/door", 0, 1, 0, 1, False, False)]})
    assert oim.to_records(sel, {"/m/door": "Door"}, "validation", tmp_path, LabelMap.load()) == []


def test_cli_refuses_the_large_train_csv_without_approval(tmp_path):
    import subprocess
    import sys

    out = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[2] / "scripts" / "acquire_data.py"),  # noqa: S603
                          "openimages", "--split", "train", "--root", str(tmp_path)],
                         capture_output=True, text=True, timeout=60, check=False)
    assert out.returncode == 2 and "Refusing" in out.stdout and not (tmp_path / "annotations").exists()


def test_group_of_and_depiction_boxes_do_not_count_toward_the_cap(tmp_path):
    rows = [("crowd", "/m/door", True, False), ("drawing", "/m/door", False, True), ("real", "/m/door", False, False)]
    sel = oim.select_images(oim.iter_bbox_rows(_bbox(tmp_path / "b.csv", rows), {"/m/door": "Door"}), 10)
    assert set(sel.images) == {"real"} and sel.class_counts["/m/door"] == 1


def test_label_map_ignore_list_becomes_ignore_regions_and_is_not_counted(tmp_path):
    lm = LabelMap.load()
    assert lm.is_ignored("openimages", "Desk")
    rows = [("img1", "/m/door", False, False), ("img1", "/m/desk", False, False), ("desk_only", "/m/desk", False, False)]
    wanted = {"/m/door": "Door", "/m/desk": "Desk"}
    sel = oim.select_images(oim.iter_bbox_rows(_bbox(tmp_path / "b.csv", rows), wanted), 10,
                            ignore_ids=frozenset({"/m/desk"}))
    assert set(sel.images) == {"img1"} and "/m/desk" not in sel.class_counts
    (tmp_path / "images" / "validation").mkdir(parents=True)
    Image.new("RGB", (200, 100)).save(tmp_path / "images" / "validation" / "img1.jpg")
    (rec,) = oim.to_records(sel, wanted, "validation", tmp_path, lm)
    assert [b.cls for b in rec.boxes] == ["door"] and len(rec.ignore_regions) == 1
