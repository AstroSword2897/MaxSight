"""The legacy COCO splitter (still used by scripts/ops/gather_training_data.py and medallion_build.py)
must check image content: two COCO ids with byte-identical images go to the same split, and an image
whose file cannot be read cannot be cleared, so the split is refused.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ml.data.coco_dataset_splitter import create_maxsight_splits_from_coco, split_coco_dataset
from PIL import Image

N_IMAGES = 40
TWINS = (3, 27)  # COCO ids whose files hold the same bytes


def _coco(tmp_path: Path) -> tuple[Path, Path]:
    image_dir = tmp_path / "train2017"
    image_dir.mkdir()
    images, annotations = [], []
    for i in range(N_IMAGES):
        name = f"{i:012d}.jpg"
        shade = TWINS[0] if i in TWINS else i  # the twins are the same picture
        Image.new("RGB", (64, 48), color=(shade * 5 % 256, 40, 90)).save(image_dir / name, quality=95)
        images.append({"id": i, "file_name": name, "width": 64, "height": 48})
        annotations.append({"id": i, "image_id": i, "category_id": 1, "bbox": [4, 4, 20, 20]})
    ann = tmp_path / "instances.json"
    ann.write_text(json.dumps({"images": images, "annotations": annotations,
                               "categories": [{"id": 1, "name": "person"}]}))
    return ann, image_dir


def _coco_split_of(files: tuple[Path, ...]) -> dict[int, str]:
    out = {}
    for path, split in zip(files, ("train", "val", "test")):
        for img in json.loads(path.read_text())["images"]:
            out[img["id"]] = split
    return out


def _maxsight_split_of(files: tuple[Path, ...]) -> dict[int, str]:
    out = {}
    for path, split in zip(files, ("train", "val", "test")):
        for ann in json.loads(path.read_text()):
            out[ann["image_id"]] = split
    return out


@pytest.mark.parametrize("seed", range(8))
def test_coco_format_keeps_identical_images_in_one_split(tmp_path, seed):
    ann, image_dir = _coco(tmp_path)
    files = split_coco_dataset(ann, tmp_path / "out", image_dir=image_dir, seed=seed)
    split_of = _coco_split_of(files)
    assert len(split_of) == N_IMAGES
    assert split_of[TWINS[0]] == split_of[TWINS[1]]


@pytest.mark.parametrize("seed", range(8))
def test_maxsight_format_keeps_identical_images_in_one_split(tmp_path, seed):
    ann, image_dir = _coco(tmp_path)
    files = create_maxsight_splits_from_coco(ann, image_dir, tmp_path / "out", seed=seed)
    split_of = _maxsight_split_of(files)
    assert len(split_of) == N_IMAGES
    assert split_of[TWINS[0]] == split_of[TWINS[1]]


def test_ratios_are_still_honoured(tmp_path):
    ann, image_dir = _coco(tmp_path)
    files = create_maxsight_splits_from_coco(ann, image_dir, tmp_path / "out", seed=0)
    counts = {s: list(_maxsight_split_of(files).values()).count(s) for s in ("train", "val", "test")}
    assert abs(counts["train"] - 28) <= 2 and abs(counts["val"] - 6) <= 2 and abs(counts["test"] - 6) <= 2


@pytest.mark.parametrize("which", ["coco", "maxsight"])
def test_a_missing_image_file_refuses_the_split(tmp_path, which):
    ann, image_dir = _coco(tmp_path)
    (image_dir / f"{5:012d}.jpg").unlink()
    with pytest.raises(ValueError, match="cannot check the content"):
        if which == "coco":
            split_coco_dataset(ann, tmp_path / "out", image_dir=image_dir)
        else:
            create_maxsight_splits_from_coco(ann, image_dir, tmp_path / "out")


def test_maxsight_split_without_an_image_dir_is_refused(tmp_path):
    ann, _ = _coco(tmp_path)
    with pytest.raises(ValueError, match="cannot check the content"):
        create_maxsight_splits_from_coco(ann, None, tmp_path / "out")
