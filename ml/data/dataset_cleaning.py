"""Bronze → silver: validate, deduplicate, remove corrupt frames/samples.

Works on any dataset registered in the medallion layout.  All functions are
pure-Python and operate on file paths, so they run without a GPU.

Usage (programmatic)
--------------------
from ml.data.dataset_cleaning import DatasetCleaner
cleaner = DatasetCleaner(bronze_dir=Path("datasets/medallion/bronze/bdd100k"),
                          silver_dir=Path("datasets/medallion/silver/bdd100k"))
report = cleaner.clean_images(ext=["jpg", "png"])

Usage (CLI)
-----------
See scripts/ops/clean_and_preprocess.py.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Supported image and video extensions.
IMAGE_EXTS: Set[str] = {"jpg", "jpeg", "png", "bmp", "webp", "tiff", "tif"}
VIDEO_EXTS: Set[str] = {"mp4", "avi", "mov", "mkv", "webm"}
ANNOTATION_EXTS: Set[str] = {"json"}


# ── Result dataclasses ─────────────────────────────────────────────────────────

@dataclass
class CleaningReport:
    dataset_key: str
    started_at: str = ""
    finished_at: str = ""
    total_scanned: int = 0
    kept: int = 0
    removed_corrupt: int = 0
    removed_duplicate: int = 0
    removed_too_small: int = 0
    annotation_errors: List[str] = field(default_factory=list)
    skipped_files: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_key": self.dataset_key,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_scanned": self.total_scanned,
            "kept": self.kept,
            "removed_corrupt": self.removed_corrupt,
            "removed_duplicate": self.removed_duplicate,
            "removed_too_small": self.removed_too_small,
            "annotation_errors": self.annotation_errors[:20],
            "skipped_files": self.skipped_files[:20],
        }


# ── Core cleaner ──────────────────────────────────────────────────────────────

class DatasetCleaner:
    """Validate and deduplicate files from a bronze directory into silver."""

    def __init__(
        self,
        bronze_dir: Path,
        silver_dir: Path,
        *,
        dataset_key: str = "unknown",
        min_image_side: int = 32,
        max_image_side: int = 16384,
        copy_files: bool = True,
    ) -> None:
        self.bronze_dir = Path(bronze_dir)
        self.silver_dir = Path(silver_dir)
        self.dataset_key = dataset_key
        self.min_image_side = min_image_side
        self.max_image_side = max_image_side
        self.copy_files = copy_files

    # ── Image cleaning ────────────────────────────────────────────────────────

    def clean_images(
        self,
        ext: Optional[List[str]] = None,
        subdir: str = "images",
    ) -> CleaningReport:
        """Scan bronze images, remove corrupt/duplicate/too-small, copy valid to silver."""
        from PIL import Image as PILImage

        report = CleaningReport(dataset_key=self.dataset_key)
        report.started_at = _now()
        exts = set(e.lower().lstrip(".") for e in (ext or list(IMAGE_EXTS)))

        silver_img_dir = self.silver_dir / subdir
        silver_img_dir.mkdir(parents=True, exist_ok=True)

        seen_hashes: Set[str] = set()
        for src in _iter_files(self.bronze_dir, exts):
            report.total_scanned += 1
            try:
                img = PILImage.open(src)
                img.verify()
                img = PILImage.open(src).convert("RGB")
                w, h = img.size
            except Exception as exc:
                logger.debug("Corrupt: %s — %s", src.name, exc)
                report.removed_corrupt += 1
                report.skipped_files.append(str(src))
                continue

            if w < self.min_image_side or h < self.min_image_side:
                report.removed_too_small += 1
                continue
            if w > self.max_image_side or h > self.max_image_side:
                report.removed_too_small += 1
                continue

            h_val = _file_hash(src)
            if h_val in seen_hashes:
                report.removed_duplicate += 1
                continue
            seen_hashes.add(h_val)

            if self.copy_files:
                _copy_file(src, silver_img_dir / src.name)
            report.kept += 1

        report.finished_at = _now()
        logger.info(
            "Image clean [%s]: scanned=%d kept=%d corrupt=%d dup=%d small=%d",
            self.dataset_key,
            report.total_scanned,
            report.kept,
            report.removed_corrupt,
            report.removed_duplicate,
            report.removed_too_small,
        )
        return report

    # ── Video cleaning ────────────────────────────────────────────────────────

    def clean_videos(
        self,
        ext: Optional[List[str]] = None,
        subdir: str = "videos",
        min_duration_s: float = 0.5,
    ) -> CleaningReport:
        """Scan bronze video files: verify openable, deduplicate, copy to silver."""
        try:
            import cv2  # type: ignore
        except ImportError:
            raise ImportError("opencv-python is required for video cleaning: pip install opencv-python")

        report = CleaningReport(dataset_key=self.dataset_key)
        report.started_at = _now()
        exts = set(e.lower().lstrip(".") for e in (ext or list(VIDEO_EXTS)))

        silver_vid_dir = self.silver_dir / subdir
        silver_vid_dir.mkdir(parents=True, exist_ok=True)

        seen_hashes: Set[str] = set()
        for src in _iter_files(self.bronze_dir, exts):
            report.total_scanned += 1
            try:
                cap = cv2.VideoCapture(str(src))
                if not cap.isOpened():
                    raise ValueError("Cannot open")
                fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
                nf = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
                cap.release()
                duration = nf / fps
                if duration < min_duration_s:
                    report.removed_too_small += 1
                    continue
            except Exception as exc:
                logger.debug("Corrupt video: %s — %s", src.name, exc)
                report.removed_corrupt += 1
                report.skipped_files.append(str(src))
                continue

            h_val = _file_hash(src, chunk_bytes=1 << 20)
            if h_val in seen_hashes:
                report.removed_duplicate += 1
                continue
            seen_hashes.add(h_val)

            if self.copy_files:
                _copy_file(src, silver_vid_dir / src.name)
            report.kept += 1

        report.finished_at = _now()
        logger.info(
            "Video clean [%s]: scanned=%d kept=%d corrupt=%d dup=%d short=%d",
            self.dataset_key,
            report.total_scanned,
            report.kept,
            report.removed_corrupt,
            report.removed_duplicate,
            report.removed_too_small,
        )
        return report

    # ── Annotation cleaning ───────────────────────────────────────────────────

    def clean_coco_annotations(
        self,
        src_annotation: Path,
        out_annotation: Path,
        *,
        drop_crowd: bool = True,
        min_box_area: float = 100.0,
    ) -> CleaningReport:
        """Clean a COCO-style annotation JSON: drop crowd, tiny boxes, missing images."""
        report = CleaningReport(dataset_key=self.dataset_key)
        report.started_at = _now()
        with open(src_annotation, encoding="utf-8") as f:
            data = json.load(f)

        valid_img_ids: Set[int] = set()
        kept_images = []
        for img in data.get("images", []):
            fn = img.get("file_name", "")
            if not fn:
                report.annotation_errors.append(f"Image missing file_name: {img.get('id')}")
                continue
            valid_img_ids.add(img["id"])
            kept_images.append(img)

        kept_anns = []
        for ann in data.get("annotations", []):
            report.total_scanned += 1
            if ann.get("image_id") not in valid_img_ids:
                report.removed_corrupt += 1
                continue
            if drop_crowd and ann.get("iscrowd", 0):
                report.removed_corrupt += 1
                continue
            box = ann.get("bbox", [])
            if len(box) == 4:
                area = box[2] * box[3]
                if area < min_box_area:
                    report.removed_too_small += 1
                    continue
            kept_anns.append(ann)
            report.kept += 1

        out_data = {
            "info": data.get("info", {}),
            "licenses": data.get("licenses", []),
            "categories": data.get("categories", []),
            "images": kept_images,
            "annotations": kept_anns,
        }
        out_annotation.parent.mkdir(parents=True, exist_ok=True)
        out_annotation.write_text(json.dumps(out_data), encoding="utf-8")
        report.finished_at = _now()
        logger.info(
            "Annotation clean [%s]: scanned=%d kept=%d removed=%d errors=%d",
            self.dataset_key,
            report.total_scanned,
            report.kept,
            report.removed_corrupt + report.removed_too_small,
            len(report.annotation_errors),
        )
        return report


# ── Dataset-specific cleaning functions ───────────────────────────────────────

def clean_coco(bronze_dir: Path, silver_dir: Path) -> List[CleaningReport]:
    """COCO: clean images and instances_*.json annotation files."""
    reports = []
    cleaner = DatasetCleaner(bronze_dir, silver_dir / "images", dataset_key="coco")
    reports.append(cleaner.clean_images())
    ann_dir = bronze_dir / "annotations"
    if ann_dir.exists():
        silver_ann = silver_dir / "annotations"
        silver_ann.mkdir(parents=True, exist_ok=True)
        ann_cleaner = DatasetCleaner(bronze_dir, silver_dir, dataset_key="coco")
        for ann_file in ann_dir.glob("instances_*.json"):
            out = silver_ann / ann_file.name
            reports.append(ann_cleaner.clean_coco_annotations(ann_file, out))
    return reports


def clean_bdd100k(bronze_dir: Path, silver_dir: Path) -> List[CleaningReport]:
    """BDD100K: images + COCO-style annotation JSON exports."""
    cleaner = DatasetCleaner(bronze_dir, silver_dir, dataset_key="bdd100k")
    reports = [cleaner.clean_images(ext=["jpg", "jpeg"])]
    ann_dir = bronze_dir / "labels"
    if ann_dir.exists():
        silver_ann = silver_dir / "annotations"
        silver_ann.mkdir(parents=True, exist_ok=True)
        ann_cleaner = DatasetCleaner(bronze_dir, silver_dir, dataset_key="bdd100k")
        for ann_file in ann_dir.glob("*.json"):
            out = silver_ann / ann_file.name
            try:
                reports.append(ann_cleaner.clean_coco_annotations(ann_file, out))
            except Exception as exc:
                logger.warning("BDD100K annotation skip %s: %s", ann_file.name, exc)
    return reports


def clean_video_dataset(
    bronze_dir: Path,
    silver_dir: Path,
    dataset_key: str,
    min_duration_s: float = 0.5,
) -> List[CleaningReport]:
    """Generic video-only cleaner for Kinetics, HowTo100M, WebVid, Epic-Kitchens."""
    cleaner = DatasetCleaner(bronze_dir, silver_dir, dataset_key=dataset_key)
    return [cleaner.clean_videos(min_duration_s=min_duration_s)]


def clean_vos_dataset(
    bronze_dir: Path,
    silver_dir: Path,
    dataset_key: str,
) -> List[CleaningReport]:
    """Video Object Segmentation datasets (MOSE, YouTube-VOS): validate frame dirs."""
    report = CleaningReport(dataset_key=dataset_key)
    report.started_at = _now()
    from PIL import Image as PILImage

    silver_frames = silver_dir / "frames"
    silver_frames.mkdir(parents=True, exist_ok=True)
    seen_hashes: Set[str] = set()

    for src in _iter_files(bronze_dir, IMAGE_EXTS):
        report.total_scanned += 1
        try:
            img = PILImage.open(src)
            img.verify()
            img = PILImage.open(src)
            w, h = img.size
            if w < 32 or h < 32:
                report.removed_too_small += 1
                continue
        except Exception:
            report.removed_corrupt += 1
            continue
        h_val = _file_hash(src)
        if h_val in seen_hashes:
            report.removed_duplicate += 1
            continue
        seen_hashes.add(h_val)
        rel = src.relative_to(bronze_dir)
        dest = silver_frames / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        _copy_file(src, dest)
        report.kept += 1

    report.finished_at = _now()
    logger.info("VOS clean [%s]: kept=%d / scanned=%d", dataset_key, report.kept, report.total_scanned)
    return [report]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _iter_files(root: Path, exts: Set[str]):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower().lstrip(".") in exts:
            yield p


def _file_hash(path: Path, chunk_bytes: int = 65536) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_bytes):
            h.update(chunk)
    return h.hexdigest()


def _copy_file(src: Path, dst: Path) -> None:
    import shutil
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")
