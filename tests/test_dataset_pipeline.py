"""Tests for bronze→silver cleaning and preprocessing (no GPU, no network)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.data.dataset_cleaning import (  # noqa: E402
    DatasetCleaner,
    clean_coco,
)
from ml.data.dataset_preprocessing import (  # noqa: E402
    ImagePreprocessingPipeline,
    VideoFrameExtractor,
    adapt_bdd100k_to_coco,
    build_vos_coco_annotation,
)
from ml.data.medallion_layout import (  # noqa: E402
    DATASET_KEYS,
    bronze_dataset_dir,
    ensure_medallion_dirs,
    load_ingest_record,
    silver_dataset_dir,
    write_ingest_record,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_images(directory: Path, count: int = 4, size: tuple = (128, 128)) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(count):
        p = directory / f"img_{i:03d}.jpg"
        Image.new("RGB", size, color=(i * 20, 100, 200)).save(p, format="JPEG")
        paths.append(p)
    return paths


def _tiny_coco_ann(images: list[Path], out_path: Path, img_dir: Path) -> None:
    imgs = [{"id": i + 1, "file_name": str(p.relative_to(img_dir)), "width": 128, "height": 128}
            for i, p in enumerate(images)]
    anns = [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 40, 40], "area": 1600, "iscrowd": 0}]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "info": {}, "licenses": [],
        "categories": [{"id": 1, "name": "person"}],
        "images": imgs, "annotations": anns,
    }), encoding="utf-8")


# ── Medallion layout tests ────────────────────────────────────────────────────

def test_dataset_keys_complete() -> None:
    expected = {"coco", "kinetics700", "youtube8m", "howto100m", "webvid10m",
                "bdd100k", "epic_kitchens", "mose", "youtube_vos"}
    assert set(DATASET_KEYS) == expected


def test_ensure_medallion_dirs(tmp_path: Path) -> None:
    ensure_medallion_dirs(tmp_path, datasets=["coco", "bdd100k"])
    assert (tmp_path / "bronze" / "coco").exists()
    assert (tmp_path / "silver" / "bdd100k" / "manifests").exists()
    assert (tmp_path / "gold").exists()


def test_ingest_record_roundtrip(tmp_path: Path) -> None:
    ensure_medallion_dirs(tmp_path, datasets=["bdd100k"])
    rec = {"dataset_key": "bdd100k", "source_path": "/data/bdd", "probe": {"ready": True}}
    write_ingest_record(tmp_path, "bdd100k", rec)
    loaded = load_ingest_record(tmp_path, "bdd100k")
    assert loaded["source_path"] == "/data/bdd"


def test_ingest_record_missing_raises(tmp_path: Path) -> None:
    ensure_medallion_dirs(tmp_path, datasets=["mose"])
    with pytest.raises(FileNotFoundError):
        load_ingest_record(tmp_path, "mose")


# ── DatasetCleaner image tests ────────────────────────────────────────────────

def test_clean_images_valid(tmp_path: Path) -> None:
    bronze = tmp_path / "bronze"
    silver = tmp_path / "silver"
    imgs = _make_images(bronze, count=4)
    cleaner = DatasetCleaner(bronze, silver, dataset_key="coco")
    report = cleaner.clean_images()
    assert report.kept == 4
    assert report.removed_corrupt == 0
    assert report.removed_duplicate == 0


def test_clean_images_deduplication(tmp_path: Path) -> None:
    bronze = tmp_path / "bronze"
    silver = tmp_path / "silver"
    bronze.mkdir(parents=True)
    img_path = bronze / "a.jpg"
    Image.new("RGB", (64, 64), color=(10, 20, 30)).save(img_path, format="JPEG")
    import shutil
    shutil.copy2(img_path, bronze / "b.jpg")
    cleaner = DatasetCleaner(bronze, silver, dataset_key="coco")
    report = cleaner.clean_images()
    assert report.kept == 1
    assert report.removed_duplicate == 1


def test_clean_images_corrupt(tmp_path: Path) -> None:
    bronze = tmp_path / "bronze"
    bronze.mkdir(parents=True)
    bad = bronze / "corrupt.jpg"
    bad.write_bytes(b"not an image at all")
    cleaner = DatasetCleaner(bronze, tmp_path / "silver", dataset_key="coco")
    report = cleaner.clean_images()
    assert report.removed_corrupt == 1
    assert report.kept == 0


def test_clean_images_too_small(tmp_path: Path) -> None:
    bronze = tmp_path / "bronze"
    bronze.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(0, 0, 0)).save(bronze / "tiny.jpg", format="JPEG")
    cleaner = DatasetCleaner(bronze, tmp_path / "silver", dataset_key="coco", min_image_side=32)
    report = cleaner.clean_images()
    assert report.removed_too_small == 1


# ── COCO annotation cleaning tests ───────────────────────────────────────────

def test_clean_coco_annotations(tmp_path: Path) -> None:
    img_dir = tmp_path / "images"
    imgs = _make_images(img_dir, count=3)
    ann_src = tmp_path / "ann_src.json"
    ann_out = tmp_path / "ann_out.json"
    _tiny_coco_ann(imgs, ann_src, img_dir)

    cleaner = DatasetCleaner(tmp_path, tmp_path, dataset_key="coco")
    report = cleaner.clean_coco_annotations(ann_src, ann_out, min_box_area=1.0)
    assert ann_out.exists()
    data = json.loads(ann_out.read_text())
    assert len(data["annotations"]) >= 1
    assert report.kept >= 1


def test_clean_coco_removes_tiny_boxes(tmp_path: Path) -> None:
    img_dir = tmp_path / "images"
    imgs = _make_images(img_dir, count=1)
    ann_src = tmp_path / "ann.json"
    ann_out = tmp_path / "ann_out.json"
    ann_src.write_text(json.dumps({
        "info": {}, "licenses": [], "categories": [{"id": 1, "name": "obj"}],
        "images": [{"id": 1, "file_name": imgs[0].name, "width": 128, "height": 128}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 2, 2], "area": 4, "iscrowd": 0},
        ],
    }), encoding="utf-8")
    cleaner = DatasetCleaner(tmp_path, tmp_path, dataset_key="coco")
    report = cleaner.clean_coco_annotations(ann_src, ann_out, min_box_area=100.0)
    data = json.loads(ann_out.read_text())
    assert len(data["annotations"]) == 0
    assert report.removed_too_small == 1


# ── Image preprocessing tests ─────────────────────────────────────────────────

def test_image_preprocessing_resize(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _make_images(src, count=3, size=(256, 256))
    pipe = ImagePreprocessingPipeline(target_size=(224, 224), dataset_key="test")
    report = pipe.run(src, dst)
    assert report.processed == 3
    out_imgs = list(dst.glob("*.jpg"))
    assert len(out_imgs) == 3
    img = Image.open(out_imgs[0])
    assert img.size == (224, 224)


def test_image_preprocessing_skip_existing(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _make_images(src, count=2, size=(128, 128))
    pipe = ImagePreprocessingPipeline(target_size=(64, 64), dataset_key="test")
    r1 = pipe.run(src, dst)
    r2 = pipe.run(src, dst)
    assert r1.processed == 2
    assert r2.skipped_existing == 2


# ── BDD100K → COCO adapter ───────────────────────────────────────────────────

def test_adapt_bdd100k_to_coco(tmp_path: Path) -> None:
    bdd = [{"name": "frame_0001.jpg", "labels": [
        {"category": "car", "box2d": {"x1": 10, "y1": 20, "x2": 100, "y2": 80}},
        {"category": "person", "box2d": {"x1": 200, "y1": 100, "x2": 250, "y2": 300}},
    ]}]
    src = tmp_path / "bdd_det.json"
    src.write_text(json.dumps(bdd), encoding="utf-8")
    out = tmp_path / "coco_bdd.json"
    result = adapt_bdd100k_to_coco(src, out)
    assert result["images"] == 1
    assert result["annotations"] == 2
    coco = json.loads(out.read_text())
    assert len(coco["images"]) == 1
    assert len(coco["annotations"]) == 2


# ── VOS scaffold ──────────────────────────────────────────────────────────────

def test_build_vos_coco_annotation(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    for vid in ["vid001", "vid002"]:
        d = frames_dir / vid
        d.mkdir(parents=True)
        for i in range(3):
            Image.new("RGB", (64, 64)).save(d / f"frame_{i:04d}.jpg", format="JPEG")
    out = tmp_path / "scaffold.json"
    result = build_vos_coco_annotation(frames_dir, None, out, dataset_key="mose")
    assert result["images"] == 6
    data = json.loads(out.read_text())
    assert len(data["images"]) == 6
