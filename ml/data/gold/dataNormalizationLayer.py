"""Adapters emit partial gold records (geometry + raw class names); mapping runs in the builder."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

from ml.data.gold.errors import GoldConfigError
from ml.data.gold.schema import GOLD_LINE_SCHEMA_VERSION, LABEL_SPACE_ACCESSIBILITY_622
from ml.data.medallion_layout import path_relative_to_repo
from ml.models.maxsight_cnn import COCO_CLASSES


def _distance_zone_from_area(area: float) -> int:
    if area > 0.1:
        return 0
    if area > 0.05:
        return 1
    return 2


def _urgency_from_category(category_name: str) -> int:
    keywords = (
        "car",
        "truck",
        "bus",
        "vehicle",
        "fire",
        "hazard",
        "stop",
        "traffic",
    )
    lower = category_name.lower()
    return 3 if any(kw in lower for kw in keywords) else 0


class MaxSightListAdapter:
    """List-style MaxSight JSON (one dict per image with ``objects``)."""

    def __init__(
        self,
        annotation_path: Path,
        image_root: Path,
        repo_root: Path,
        *,
        dataset_id: str,
        version: str,
        split: str,
    ) -> None:
        self.annotation_path = Path(annotation_path)
        self.image_root = Path(image_root).resolve()
        self.repo_root = Path(repo_root).resolve()
        self.dataset_id = dataset_id
        self.version = version
        self.split = split
        self._source_file = path_relative_to_repo(
            self.annotation_path.resolve(), self.repo_root
        )
        with self.annotation_path.open(encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(
                f"MaxSight list adapter expects a JSON array, got {type(data).__name__}"
            )
        self._rows: List[Dict[str, Any]] = data

    def __len__(self) -> int:
        return len(self._rows)

    def load_partial(self, idx: int) -> Dict[str, Any]:
        return self._partial_for_row(self._rows[idx])

    def _partial_for_row(self, ann: Dict[str, Any]) -> Dict[str, Any]:
        image_id = ann.get("image_id", ann.get("id", 0))
        raw_path = ann.get("image_path", f"{image_id}.jpg")
        p = Path(raw_path)
        abs_img = p.resolve() if p.is_absolute() else (self.image_root / raw_path).resolve()
        width, height = 0, 0
        if abs_img.is_file():
            try:
                with Image.open(abs_img) as im:
                    width, height = im.size
            except Exception:
                width, height = 0, 0
        label_names: List[str] = []
        boxes: List[List[float]] = []
        distances: List[int] = []
        object_urgencies: List[int] = []
        for obj in ann.get("objects", []):
            category = obj.get("category")
            if isinstance(category, str):
                name = category
            else:
                ci = int(obj.get("class", 0))
                name = COCO_CLASSES[min(max(ci, 0), len(COCO_CLASSES) - 1)]
            label_names.append(name)
            box = obj.get("box", [0.5, 0.5, 0.1, 0.1])
            cx, cy, w, h = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            w = max(1e-4, min(1.0, w))
            h = max(1e-4, min(1.0, h))
            area = w * h
            boxes.append([cx, cy, w, h])
            distances.append(_distance_zone_from_area(area))
            object_urgencies.append(_urgency_from_category(name))
        scene_urgency = int(ann.get("urgency", 0))
        if object_urgencies:
            scene_urgency = max(scene_urgency, max(object_urgencies))
        lighting = ann.get("lighting", "normal")
        rel_image = path_relative_to_repo(abs_img, self.repo_root)
        return {
            "schema_version": GOLD_LINE_SCHEMA_VERSION,
            "image_path": rel_image,
            "boxes": boxes,
            "label_names": label_names,
            "distances": distances,
            "object_urgencies": object_urgencies,
            "scene_urgency": scene_urgency,
            "metadata": {
                "dataset_id": self.dataset_id,
                "split": self.split,
                "label_space": LABEL_SPACE_ACCESSIBILITY_622,
                "source_format": "maxsight_list",
                "version": self.version,
                "conditions": {"lighting": str(lighting)},
                "source_file": self._source_file,
                "original_id": str(image_id),
                "width": int(width),
                "height": int(height),
            },
        }


class COCOAdapter:
    """COCO detection JSON (``images`` + ``annotations``), instance bboxes only."""

    def __init__(
        self,
        coco_annotation_path: Path,
        image_root: Path,
        repo_root: Path,
        *,
        dataset_id: str,
        version: str,
        split: str,
    ) -> None:
        self.coco_annotation_path = Path(coco_annotation_path)
        self.image_root = Path(image_root).resolve()
        self.repo_root = Path(repo_root).resolve()
        self.dataset_id = dataset_id
        self.version = version
        self.split = split
        self._source_file = path_relative_to_repo(
            self.coco_annotation_path.resolve(), self.repo_root
        )
        with self.coco_annotation_path.open(encoding="utf-8") as f:
            data = json.load(f)
        images = data.get("images") or []
        anns_raw = data.get("annotations") or []
        self._partials: List[Dict[str, Any]] = []
        if not images or not anns_raw:
            return
        first = anns_raw[0] if anns_raw else {}
        if "segments_info" in first:
            return
        image_map = {img["id"]: img for img in images}
        category_map = {c["id"]: c["name"] for c in data.get("categories", [])}
        by_image: Dict[int, List[Dict[str, Any]]] = {}
        for ann in anns_raw:
            if "bbox" not in ann:
                continue
            iid = ann["image_id"]
            by_image.setdefault(iid, []).append(ann)
        for image_id in sorted(by_image.keys()):
            img_info = image_map.get(image_id)
            if not img_info:
                continue
            iw = float(img_info.get("width") or 1.0)
            ih = float(img_info.get("height") or 1.0)
            width = int(img_info.get("width") or 0)
            height = int(img_info.get("height") or 0)
            file_name = img_info.get("file_name") or f"{image_id}.jpg"
            abs_img = (self.image_root / file_name).resolve()
            label_names: List[str] = []
            boxes: List[List[float]] = []
            distances: List[int] = []
            object_urgencies: List[int] = []
            for ann in by_image[image_id]:
                bbox = ann.get("bbox", [0.0, 0.0, 0.0, 0.0])
                bx, by, bw, bh = (
                    float(bbox[0]),
                    float(bbox[1]),
                    max(1e-3, float(bbox[2])),
                    max(1e-3, float(bbox[3])),
                )
                cx = (bx + bw / 2.0) / max(1.0, iw)
                cy = (by + bh / 2.0) / max(1.0, ih)
                w = bw / max(1.0, iw)
                h = bh / max(1.0, ih)
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                w = max(1e-4, min(1.0, w))
                h = max(1e-4, min(1.0, h))
                cat_name = category_map.get(ann.get("category_id", 0), "unknown")
                label_names.append(str(cat_name))
                area = w * h
                boxes.append([cx, cy, w, h])
                distances.append(_distance_zone_from_area(area))
                object_urgencies.append(_urgency_from_category(cat_name))
            scene_urgency = max(object_urgencies) if object_urgencies else 0
            self._partials.append(
                {
                    "schema_version": GOLD_LINE_SCHEMA_VERSION,
                    "image_path": path_relative_to_repo(abs_img, self.repo_root),
                    "boxes": boxes,
                    "label_names": label_names,
                    "distances": distances,
                    "object_urgencies": object_urgencies,
                    "scene_urgency": scene_urgency,
                    "metadata": {
                        "dataset_id": self.dataset_id,
                        "split": self.split,
                        "label_space": LABEL_SPACE_ACCESSIBILITY_622,
                        "source_format": "coco_instances",
                        "version": self.version,
                        "conditions": {"lighting": "normal"},
                        "source_file": self._source_file,
                        "original_id": str(image_id),
                        "width": width,
                        "height": height,
                    },
                }
            )

    def __len__(self) -> int:
        return len(self._partials)

    def load_partial(self, idx: int) -> Dict[str, Any]:
        return dict(self._partials[idx])


class VideoManifestAdapter:
    """Reserved: frame manifests into gold lines (not implemented)."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise GoldConfigError(
            "Video manifest → gold is not implemented; use data_plane legacy with "
            "video loaders or extend VideoManifestAdapter first."
        )
