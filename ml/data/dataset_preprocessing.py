"""Silver → silver: resize, normalise images; extract frames from video clips.

All functions write deterministically into the silver layer so re-running is
safe (existing files are skipped unless force=True).

Usage (programmatic)
--------------------
from ml.data.dataset_preprocessing import ImagePreprocessingPipeline, VideoFrameExtractor

pipe = ImagePreprocessingPipeline(target_size=(224, 224))
pipe.run(silver_images_dir, output_dir)

ext = VideoFrameExtractor(fps=1.0, target_size=(224, 224))
ext.extract(video_path, frames_output_dir)

Usage (CLI)
-----------
See scripts/ops/clean_and_preprocess.py.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ImageNet normalisation (default for all model inputs).
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class PreprocessingReport:
    dataset_key: str
    stage: str
    started_at: str = ""
    finished_at: str = ""
    processed: int = 0
    skipped_existing: int = 0
    errors: int = 0
    error_samples: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_key": self.dataset_key,
            "stage": self.stage,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "processed": self.processed,
            "skipped_existing": self.skipped_existing,
            "errors": self.errors,
            "error_samples": self.error_samples[:20],
        }


# ── Image preprocessing ───────────────────────────────────────────────────────

class ImagePreprocessingPipeline:
    """Resize and optionally normalise images into a flat output directory."""

    def __init__(
        self,
        *,
        target_size: Tuple[int, int] = (224, 224),
        normalise: bool = False,
        output_format: str = "JPEG",
        jpeg_quality: int = 95,
        dataset_key: str = "unknown",
    ) -> None:
        self.target_size = target_size  # (W, H)
        self.normalise = normalise
        self.output_format = output_format.upper()
        self.jpeg_quality = jpeg_quality
        self.dataset_key = dataset_key

    def run(
        self,
        src_dir: Path,
        out_dir: Path,
        *,
        extensions: Optional[Set[str]] = None,
        force: bool = False,
    ) -> PreprocessingReport:
        from PIL import Image

        exts = extensions or {"jpg", "jpeg", "png", "bmp", "webp"}
        out_dir.mkdir(parents=True, exist_ok=True)
        report = PreprocessingReport(dataset_key=self.dataset_key, stage="image_resize")
        report.started_at = _now()
        suffix = ".jpg" if self.output_format == "JPEG" else f".{self.output_format.lower()}"

        for src in _iter_files(src_dir, exts):
            out_path = out_dir / (src.stem + suffix)
            if out_path.exists() and not force:
                report.skipped_existing += 1
                continue
            try:
                img = Image.open(src).convert("RGB")
                img = img.resize(self.target_size, Image.BILINEAR)
                if self.normalise:
                    arr = np.array(img, dtype=np.float32) / 255.0
                    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
                    std = np.array(IMAGENET_STD, dtype=np.float32)
                    arr = (arr - mean) / std
                    # Clip back to [0,1] for saving as PIL.
                    arr = np.clip(arr, 0.0, 1.0)
                    img = Image.fromarray((arr * 255).astype(np.uint8))
                save_kwargs: Dict[str, Any] = {}
                if self.output_format == "JPEG":
                    save_kwargs["quality"] = self.jpeg_quality
                img.save(out_path, format=self.output_format, **save_kwargs)
                report.processed += 1
            except Exception as exc:
                logger.debug("Preprocess error %s: %s", src.name, exc)
                report.errors += 1
                report.error_samples.append(str(src))

        report.finished_at = _now()
        logger.info(
            "Image preprocess [%s]: processed=%d skipped=%d errors=%d",
            self.dataset_key, report.processed, report.skipped_existing, report.errors,
        )
        return report


# ── Video frame extraction ────────────────────────────────────────────────────

class VideoFrameExtractor:
    """Extract frames from video files at a target FPS into per-video subdirs."""

    def __init__(
        self,
        *,
        fps: float = 1.0,
        target_size: Optional[Tuple[int, int]] = (224, 224),
        output_format: str = "JPEG",
        jpeg_quality: int = 95,
        dataset_key: str = "unknown",
        max_frames_per_video: Optional[int] = None,
    ) -> None:
        self.fps = fps
        self.target_size = target_size
        self.output_format = output_format.upper()
        self.jpeg_quality = jpeg_quality
        self.dataset_key = dataset_key
        self.max_frames_per_video = max_frames_per_video

    def extract(
        self,
        video_path: Path,
        frames_dir: Path,
        *,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Extract frames from a single video; return frame paths and metadata."""
        try:
            import cv2  # type: ignore
        except ImportError:
            raise ImportError("opencv-python required: pip install opencv-python")
        from PIL import Image

        frames_dir.mkdir(parents=True, exist_ok=True)
        suffix = ".jpg" if self.output_format == "JPEG" else f".{self.output_format.lower()}"

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, round(src_fps / self.fps))
        frame_paths: List[str] = []
        frame_idx = 0
        saved = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                if self.max_frames_per_video and saved >= self.max_frames_per_video:
                    break
                out_name = f"frame_{saved:06d}{suffix}"
                out_path = frames_dir / out_name
                if out_path.exists() and not force:
                    frame_paths.append(str(out_path))
                    saved += 1
                    frame_idx += 1
                    continue
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                if self.target_size:
                    pil_img = pil_img.resize(self.target_size, Image.BILINEAR)
                save_kwargs: Dict[str, Any] = {}
                if self.output_format == "JPEG":
                    save_kwargs["quality"] = self.jpeg_quality
                pil_img.save(out_path, format=self.output_format, **save_kwargs)
                frame_paths.append(str(out_path))
                saved += 1
            frame_idx += 1

        cap.release()
        return {
            "video": str(video_path),
            "frames_dir": str(frames_dir),
            "frame_paths": frame_paths,
            "total_frames_extracted": saved,
            "extraction_fps": self.fps,
        }

    def run_batch(
        self,
        src_dir: Path,
        frames_root: Path,
        *,
        video_exts: Optional[Set[str]] = None,
        force: bool = False,
    ) -> PreprocessingReport:
        """Extract frames from all videos under src_dir."""
        exts = video_exts or {"mp4", "avi", "mov", "mkv", "webm"}
        report = PreprocessingReport(dataset_key=self.dataset_key, stage="video_frame_extract")
        report.started_at = _now()

        for vpath in _iter_files(src_dir, exts):
            vid_dir = frames_root / vpath.stem
            if vid_dir.exists() and any(vid_dir.iterdir()) and not force:
                report.skipped_existing += 1
                continue
            try:
                self.extract(vpath, vid_dir, force=force)
                report.processed += 1
            except Exception as exc:
                logger.warning("Frame extract error %s: %s", vpath.name, exc)
                report.errors += 1
                report.error_samples.append(str(vpath))

        report.finished_at = _now()
        logger.info(
            "Frame extraction [%s]: extracted=%d skipped=%d errors=%d",
            self.dataset_key, report.processed, report.skipped_existing, report.errors,
        )
        return report


# ── COCO annotation adapter ───────────────────────────────────────────────────

def adapt_bdd100k_to_coco(
    bdd_json_path: Path,
    out_coco_path: Path,
    *,
    category_map: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Convert BDD100K detection JSON to COCO-format annotation file.

    BDD100K detection format has top-level list of frames each with
    ``labels`` list containing ``box2d`` and ``category``.
    """
    with open(bdd_json_path, encoding="utf-8") as f:
        bdd = json.load(f)

    if category_map is None:
        category_map = {
            "car": 3, "truck": 8, "bus": 6, "person": 1, "rider": 1,
            "bicycle": 2, "motorcycle": 4, "traffic light": 10,
            "traffic sign": 10, "train": 7,
        }

    categories = [{"id": v, "name": k} for k, v in sorted(set(
        (k, v) for k, v in category_map.items()
    ), key=lambda x: x[1])]

    images, annotations = [], []
    img_id = 1
    ann_id = 1

    frames = bdd if isinstance(bdd, list) else bdd.get("frames", [])
    for frame in frames:
        fname = frame.get("name", f"{img_id:08d}.jpg")
        images.append({
            "id": img_id,
            "file_name": fname,
            "width": 1280,
            "height": 720,
        })
        for label in frame.get("labels", []):
            cat = label.get("category", "")
            cat_id = category_map.get(cat.lower(), 0)
            if cat_id == 0:
                continue
            box2d = label.get("box2d", {})
            x1 = box2d.get("x1", 0)
            y1 = box2d.get("y1", 0)
            x2 = box2d.get("x2", 1)
            y2 = box2d.get("y2", 1)
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            if w < 1 or h < 1:
                continue
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [x1, y1, w, h],
                "area": float(w * h),
                "iscrowd": 0,
            })
            ann_id += 1
        img_id += 1

    coco = {
        "info": {"description": "BDD100K → COCO adapter", "source": str(bdd_json_path)},
        "categories": categories,
        "images": images,
        "annotations": annotations,
    }
    out_coco_path.parent.mkdir(parents=True, exist_ok=True)
    out_coco_path.write_text(json.dumps(coco), encoding="utf-8")
    logger.info("BDD100K → COCO: %d images, %d annotations → %s", len(images), len(annotations), out_coco_path)
    return {"images": len(images), "annotations": len(annotations)}


def build_vos_coco_annotation(
    frames_dir: Path,
    masks_dir: Optional[Path],
    out_coco_path: Path,
    dataset_key: str = "vos",
) -> Dict[str, Any]:
    """Build a minimal COCO-style JSON for VOS datasets (MOSE / YouTube-VOS).

    Each video's frames become images; mask presence is noted but not converted
    (mask→bbox conversion requires full parsing; this creates the scaffolding).
    """
    images = []
    img_id = 1
    for video_dir in sorted(frames_dir.iterdir()):
        if not video_dir.is_dir():
            continue
        for frame in sorted(video_dir.glob("*.jpg")) + sorted(video_dir.glob("*.png")):
            images.append({
                "id": img_id,
                "file_name": str(frame.relative_to(frames_dir)),
                "video_id": video_dir.name,
            })
            img_id += 1

    coco: Dict[str, Any] = {
        "info": {"description": f"{dataset_key} frame scaffold", "source": str(frames_dir)},
        "categories": [{"id": 1, "name": "object"}],
        "images": images,
        "annotations": [],
    }
    out_coco_path.parent.mkdir(parents=True, exist_ok=True)
    out_coco_path.write_text(json.dumps(coco), encoding="utf-8")
    logger.info("VOS scaffold [%s]: %d images → %s", dataset_key, len(images), out_coco_path)
    return {"images": len(images)}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _iter_files(root: Path, exts: Set[str]):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower().lstrip(".") in exts:
            yield p


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")
