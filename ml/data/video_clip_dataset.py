"""Dataset over v1 video panoptic clip manifests (sequence-native samples)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ml.data.temporal_clip_targets import TemporalClipTargets, derive_temporal_clip_targets
from ml.data.video_manifest import validate_manifest_v1
from ml.models.maxsight_cnn import COCO_CLASSES
from ml.utils.preprocessing import ImagePreprocessor


def _bbox_xywh_pixels_to_cxcywh_norm(bbox: List[float], width: int, height: int) -> List[float]:
    x, y, w, h = [float(v) for v in bbox]
    w_img = max(1.0, float(width))
    h_img = max(1.0, float(height))
    cx = (x + w / 2.0) / w_img
    cy = (y + h / 2.0) / h_img
    nw = max(1e-4, min(1.0, w / w_img))
    nh = max(1e-4, min(1.0, h / h_img))
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    return [cx, cy, nw, nh]


class VideoClipManifestDataset(Dataset):
    """One sample per manifest clip: `frames` [T,3,H,W] plus detection targets from the last frame."""

    def __init__(
        self,
        manifest_path: Path,
        *,
        manifest_root: Optional[Path] = None,
        condition_mode: Optional[str] = None,
        apply_lighting_augmentation: bool = False,
        max_objects: int = 10,
        num_classes: int = len(COCO_CLASSES),
    ):
        self.manifest_path = Path(manifest_path)
        root = manifest_root if manifest_root is not None else self.manifest_path.parent
        self.manifest_root = Path(root)
        self.max_objects = max_objects
        self.num_classes = max(1, int(num_classes))
        self.preprocessor = ImagePreprocessor(condition_mode=condition_mode)
        self.apply_lighting_augmentation = apply_lighting_augmentation

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            self._data = json.load(f)
        errs = validate_manifest_v1(self._data)
        if errs:
            raise ValueError("Invalid manifest: " + "; ".join(errs[:5]))
        clips = self._data.get("clips", [])
        if not isinstance(clips, list):
            raise ValueError("manifest clips must be a list")
        self._clips: List[Dict[str, Any]] = [c for c in clips if isinstance(c, dict)]

    def __len__(self) -> int:
        return len(self._clips)

    def _resolve_path(self, p: str) -> Path:
        path = Path(p)
        if path.is_absolute():
            return path
        return (self.manifest_root / path).resolve()

    def _segment_to_class_idx(self, seg: Dict[str, Any]) -> int:
        if "class_idx" in seg:
            v = int(seg["class_idx"])
            return max(0, min(self.num_classes - 1, v))
        if "category_id" in seg:
            v = int(seg["category_id"])
            return max(0, min(self.num_classes - 1, v))
        return 0

    def _objects_from_last_frame(self, clip: Dict[str, Any]) -> List[Dict[str, Any]]:
        segs_wrap = clip.get("frames_segments")
        paths = clip.get("frame_paths")
        if not isinstance(segs_wrap, list) or not segs_wrap:
            return []
        last_idx = len(segs_wrap) - 1
        frame_segs = segs_wrap[last_idx]
        if not isinstance(frame_segs, list) or not frame_segs:
            return []

        path_str = paths[last_idx] if isinstance(paths, list) and last_idx < len(paths) else None
        if path_str is None:
            return []
        img_path = self._resolve_path(str(path_str))
        if not img_path.exists():
            w, h = 224, 224
        else:
            with Image.open(img_path) as im:
                w, h = im.size

        objects: List[Dict[str, Any]] = []
        for seg in frame_segs:
            if not isinstance(seg, dict):
                continue
            bbox = seg.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            box = _bbox_xywh_pixels_to_cxcywh_norm([float(x) for x in bbox], w, h)
            area = box[2] * box[3]
            if area > 0.1:
                dz = 0
            elif area > 0.05:
                dz = 1
            else:
                dz = 2
            objects.append(
                {
                    "box": box,
                    "class": self._segment_to_class_idx(seg),
                    "distance": dz,
                    "urgency": 0,
                }
            )
        return objects[: self.max_objects]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        clip = self._clips[idx]
        paths = clip.get("frame_paths")
        if not isinstance(paths, list):
            raise KeyError("clip missing frame_paths")

        frame_tensors: List[torch.Tensor] = []
        for p in paths:
            img_path = self._resolve_path(str(p))
            if not img_path.exists():
                image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            else:
                try:
                    image = Image.open(img_path).convert("RGB")
                except Exception:
                    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

            if self.apply_lighting_augmentation:
                preprocessed = self.preprocessor.preprocess_with_lighting(image)
                frame_tensors.append(preprocessed["image"])
            else:
                frame_tensors.append(self.preprocessor(image))

        frames = torch.stack(frame_tensors, dim=0)

        objects = self._objects_from_last_frame(clip)
        num_objs = min(len(objects), self.max_objects)
        labels = torch.zeros(self.max_objects, dtype=torch.long)
        boxes = torch.zeros(self.max_objects, 4, dtype=torch.float32)
        distance = torch.zeros(self.max_objects, dtype=torch.long)
        for i in range(num_objs):
            obj = objects[i]
            labels[i] = obj["class"]
            boxes[i] = torch.tensor(obj["box"], dtype=torch.float32)
            distance[i] = obj["distance"]
        urgency = 0 if not objects else max(o.get("urgency", 0) for o in objects)

        segs = clip.get("frames_segments")
        if isinstance(segs, list):
            tt = derive_temporal_clip_targets(segs)
        else:
            tt = TemporalClipTargets(1.0, 0.0)

        out: Dict[str, Any] = {
            "frames": frames,
            "labels": labels,
            "boxes": boxes,
            "distance": distance,
            "num_objects": torch.tensor(num_objs, dtype=torch.long),
            "urgency": torch.tensor(urgency, dtype=torch.long),
            "temporal_consistency": torch.tensor([tt.temporal_consistency], dtype=torch.float32),
            "flicker": torch.tensor([tt.flicker_proxy], dtype=torch.float32),
            "clip_id": clip.get("clip_id", str(idx)),
        }
        cm = getattr(self.preprocessor, "condition_mode", None)
        if cm:
            out["condition_mode"] = cm
        return out
