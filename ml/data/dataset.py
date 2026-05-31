"""Dataset loader with environmental context, audio, and condition-specific augmentations."""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio
from PIL import Image
from torch.utils.data import Dataset

from ml.data.assistive_supervision import load_assistive_spec, object_distance_and_urgency
from ml.models.maxsight_cnn import COCO_CLASSES
from ml.utils.preprocessing import ImagePreprocessor

logger = logging.getLogger(__name__)

_DATASET_ASSISTIVE_SPEC = load_assistive_spec()


class MaxSightDataset(Dataset):
    """Dataset for MaxSight: images, audio, annotations with condition-specific augmentations."""

    def __init__(
        self,
        data_dir: Path,
        annotation_file: Path | None = None,
        image_dir: Path | None = None,
        audio_dir: Path | None = None,
        condition_mode: str | None = None,
        tag_lighting_metadata: bool = True,
        lighting_pixel_augmentation: bool = False,
        max_objects: int = 10,
        strict_images: bool = False,
        dataset_source_key: str | None = None,
    ):

        self.data_dir = Path(data_dir)
        self.annotation_file = Path(annotation_file) if annotation_file else None
        self.image_dir = Path(image_dir) if image_dir else self.data_dir / "images"
        self.audio_dir = (
            Path(audio_dir)
            if audio_dir
            else (self.data_dir / "audio" if (self.data_dir / "audio").exists() else None)
        )
        self.condition_mode = condition_mode
        # tag_lighting_metadata: infer and attach lighting tag to sample dict
        # (dim / normal) without touching pixels. Renamed from the previous
        # apply_lighting_augmentation which implied pixel changes that never happened.
        self.tag_lighting_metadata = tag_lighting_metadata
        # lighting_pixel_augmentation: actually modify luminance/gamma to simulate
        # dim / overexposed conditions. Distinct from tagging so each knob is honest.
        # TODO: implement pixel-level transforms in ml/utils/preprocessing.py.
        self.lighting_pixel_augmentation = lighting_pixel_augmentation
        self.max_objects = max_objects
        # strict_images=True refuses to substitute random noise for a missing
        # file; production training should set it True so a path typo fails
        # fast instead of feeding the model pure noise that looks plausible.
        self.strict_images = strict_images
        self._missing_warned = False
        # Provenance tag for multi-source training; copied into each sample dict.
        self.dataset_source_key = dataset_source_key or ""

        # Initialize preprocessor with condition-specific transforms.
        self.preprocessor = ImagePreprocessor(condition_mode=condition_mode)

        # Class name to index mapping (must be defined before _load_annotations)
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(COCO_CLASSES)}
        self.idx_to_class = {idx: cls_name for idx, cls_name in enumerate(COCO_CLASSES)}

        # Load annotations.
        self.annotations = self._load_annotations()

        # Create image/annotation mapping.
        self.image_ids = list(self.annotations.keys()) if self.annotations else []

    def _load_annotations(self) -> dict[str, Any]:
        """Load annotations from JSON (COCO or custom format)."""
        if not self.annotation_file or not self.annotation_file.exists():
            return {}

        with open(self.annotation_file) as f:
            data = json.load(f)

        annotations: dict[Any, dict[str, Any]] = {}

        # Detect format: COCO-style has 'images'/'annotations'; custom format is simpler.
        if "images" in data and "annotations" in data:
            image_map = {img["id"]: img for img in data["images"]}
            category_map = {cat["id"]: cat["name"] for cat in data.get("categories", [])}

            # Panoptic COCO stores per-image segments under 'segments_info'.
            first_ann = data["annotations"][0] if data["annotations"] else {}
            is_panoptic = "segments_info" in first_ann

            # Group annotations by image_id for efficient per-image processing.
            for ann in data["annotations"]:
                image_id = ann["image_id"]
                if image_id not in annotations:
                    img_info = image_map.get(image_id, {})
                    annotations[image_id] = {
                        "image_path": self.image_dir / img_info.get("file_name", f"{image_id}.jpg"),
                        "objects": [],
                        "urgency": 0,
                        "lighting": "normal",
                        "audio_path": None,
                    }

                if is_panoptic:
                    segments = ann.get("segments_info", [])
                    for seg in segments:
                        bbox = seg.get("bbox", [0.0, 0.0, 0.0, 0.0])
                        category_id = seg.get("category_id", 0)
                        img_info = image_map.get(image_id, {})
                        img_width = img_info.get("width", 224)
                        img_height = img_info.get("height", 224)

                        bbox_x = max(0.0, float(bbox[0]))
                        bbox_y = max(0.0, float(bbox[1]))
                        bbox_w = max(1e-3, float(bbox[2]))
                        bbox_h = max(1e-3, float(bbox[3]))

                        cx = (bbox_x + bbox_w / 2) / max(1.0, img_width)
                        cy = (bbox_y + bbox_h / 2) / max(1.0, img_height)
                        w = bbox_w / max(1.0, img_width)
                        h = bbox_h / max(1.0, img_height)

                        cx = max(0.0, min(1.0, cx))
                        cy = max(0.0, min(1.0, cy))
                        w = max(1e-4, min(1.0, w))
                        h = max(1e-4, min(1.0, h))

                        category_name = category_map.get(category_id, "unknown")
                        class_idx = self.class_to_idx.get(category_name, 0)
                        distance_zone, urgency = object_distance_and_urgency(
                            cx, cy, w, h, category_name, _DATASET_ASSISTIVE_SPEC
                        )
                        annotations[image_id]["objects"].append(
                            {
                                "box": [cx, cy, w, h],
                                "class": class_idx,
                                "category": category_name,
                                "distance": distance_zone,
                                "urgency": urgency,
                            }
                        )
                        annotations[image_id]["urgency"] = max(
                            annotations[image_id]["urgency"], urgency
                        )
                else:
                    bbox = ann["bbox"]
                    img_info = image_map[image_id]
                    img_width = img_info.get("width", 224)
                    img_height = img_info.get("height", 224)

                    bbox_x = max(0, float(bbox[0]))
                    bbox_y = max(0, float(bbox[1]))
                    bbox_w = max(1e-3, float(bbox[2]))
                    bbox_h = max(1e-3, float(bbox[3]))

                    cx = (bbox_x + bbox_w / 2) / max(1.0, img_width)
                    cy = (bbox_y + bbox_h / 2) / max(1.0, img_height)
                    w = bbox_w / max(1.0, img_width)
                    h = bbox_h / max(1.0, img_height)

                    cx = max(0.0, min(1.0, cx))
                    cy = max(0.0, min(1.0, cy))
                    w = max(1e-4, min(1.0, w))
                    h = max(1e-4, min(1.0, h))

                    category_name = category_map.get(ann["category_id"], "unknown")
                    class_idx = self.class_to_idx.get(category_name, 0)
                    distance_zone, urgency = object_distance_and_urgency(
                        cx, cy, w, h, category_name, _DATASET_ASSISTIVE_SPEC
                    )
                    annotations[image_id]["objects"].append(
                        {
                            "box": [cx, cy, w, h],
                            "class": class_idx,
                            "category": category_name,
                            "distance": distance_zone,
                            "urgency": urgency,
                        }
                    )
                    annotations[image_id]["urgency"] = max(
                        annotations[image_id]["urgency"], urgency
                    )
        else:
            for ann in data:
                image_id = ann.get("image_id", ann.get("id", len(annotations)))
                resolved_objects: list[dict[str, Any]] = []
                for obj in ann.get("objects", []):
                    # Custom format carries 'category' as the canonical label; map it
                    # into the 622-way COCO_CLASSES index here so __getitem__ does
                    # not fall back to obj.get('class', 0) and silently zero the
                    # entire batch into class 'person'.
                    resolved = dict(obj)
                    if "class" not in resolved:
                        category = resolved.get("category")
                        if isinstance(category, str):
                            resolved["class"] = self.class_to_idx.get(category, 0)
                        else:
                            resolved["class"] = 0
                    resolved_objects.append(resolved)
                annotations[image_id] = {
                    "image_path": self.image_dir / ann.get("image_path", f"{image_id}.jpg"),
                    "objects": resolved_objects,
                    "urgency": ann.get("urgency", 0),
                    "lighting": ann.get("lighting", "normal"),
                    "audio_path": ann.get("audio_path"),
                }

        return annotations

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:

        image_id = self.image_ids[idx]
        ann = self.annotations[image_id]

        # Load image from file.
        image_path = ann["image_path"]
        if isinstance(image_path, str):
            image_path = Path(image_path)

        if not image_path.exists():
            if self.strict_images:
                raise FileNotFoundError(
                    f"MaxSightDataset strict_images=True: image_id={image_id} "
                    f"references {image_path} which does not exist."
                )
            if not self._missing_warned:
                logger.warning(
                    "MaxSightDataset: image file missing, substituting random pixels. "
                    "This is silent data corruption if it happens during training. "
                    "First offender: image_id=%s path=%s (further misses suppressed).",
                    image_id,
                    image_path,
                )
                self._missing_warned = True
            image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        else:
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as exc:
                if self.strict_images:
                    raise RuntimeError(
                        f"MaxSightDataset strict_images=True: image_id={image_id} "
                        f"at {image_path} failed to decode: {exc}"
                    ) from exc
                if not self._missing_warned:
                    logger.warning(
                        "MaxSightDataset: image failed to decode, substituting random pixels. "
                        "First offender: image_id=%s path=%s err=%s",
                        image_id,
                        image_path,
                        exc,
                    )
                    self._missing_warned = True
                image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

        # Preprocess image; optionally tag lighting metadata.
        if self.tag_lighting_metadata:
            # Detect and attach a lighting tag without modifying pixels.
            preprocessed = self.preprocessor.preprocess_with_lighting(image)
            image_tensor = preprocessed["image"]
            lighting = preprocessed.get("lighting", ann.get("lighting", "normal"))
        else:
            image_tensor = self.preprocessor(image)
            lighting = ann.get("lighting", "normal")

        # Apply actual pixel-level lighting augmentation when opted in.
        # TODO: implement luminance/gamma distortion transforms in
        # ml/utils/preprocessing.py and call them here.
        if self.lighting_pixel_augmentation:
            pass  # placeholder; no-op until pixel transforms are implemented

        # Load audio if available.
        audio_tensor = None
        if self.audio_dir and ann.get("audio_path"):
            audio_path = self.audio_dir / ann["audio_path"]
            if audio_path.exists():
                try:
                    # Load audio and extract features (MFCC or raw waveform)
                    waveform, sample_rate = torchaudio.load(str(audio_path))
                    # Use first channel if stereo, resample to 16kHz if needed.
                    if waveform.shape[0] > 1:
                        waveform = waveform[0:1]  # Take first channel.
                    if sample_rate != 16000:
                        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                        waveform = resampler(waveform)
                    # Extract MFCC features (13 coefficients, standard for audio)
                    mfcc_transform = torchaudio.transforms.MFCC(
                        sample_rate=16000,
                        n_mfcc=13,
                        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23},
                    )
                    audio_tensor = mfcc_transform(waveform)  # [1, 13, T].
                except Exception:
                    # Fallback on error.
                    audio_tensor = None

        # Extract objects and format targets.
        objects = ann.get("objects", [])
        num_objs = min(len(objects), self.max_objects)

        # Initialize padded arrays.
        labels = torch.zeros(self.max_objects, dtype=torch.long)
        boxes = torch.zeros(self.max_objects, 4, dtype=torch.float32)
        distance = torch.zeros(self.max_objects, dtype=torch.long)

        # Fill with actual objects.
        for i in range(num_objs):
            obj = objects[i]
            labels[i] = obj.get("class", 0)

            # Validate and clamp box coordinates.
            box = obj.get("box", [0.5, 0.5, 0.1, 0.1])
            box_tensor = torch.tensor(box, dtype=torch.float32)

            # Clamp to valid ranges: center [0, 1], size [1e-4, 1].
            box_tensor[0] = torch.clamp(box_tensor[0], 0.0, 1.0)  # Cx.
            box_tensor[1] = torch.clamp(box_tensor[1], 0.0, 1.0)  # Cy.
            box_tensor[2] = torch.clamp(box_tensor[2], 1e-4, 1.0)  # W.
            box_tensor[3] = torch.clamp(box_tensor[3], 1e-4, 1.0)  # H.

            # Check for NaN/Inf.
            if torch.isnan(box_tensor).any() or torch.isinf(box_tensor).any():
                box_tensor = torch.tensor([0.5, 0.5, 0.1, 0.1], dtype=torch.float32)

            boxes[i] = box_tensor
            distance[i] = obj.get("distance", 1)

        # Get scene urgency (max of all object urgencies)
        urgency = ann.get("urgency", 0)
        if objects:
            urgency = max(urgency, max(obj.get("urgency", 0) for obj in objects))

        # Build return dictionary.
        result = {
            "images": image_tensor,  # [3, H, W] preprocessed image.
            "labels": labels,  # [max_objects] class labels (padded)
            "boxes": boxes,
            "urgency": torch.tensor(urgency, dtype=torch.long),  # Scene urgency (0-3)
            "distance": distance,
            "num_objects": torch.tensor(num_objs, dtype=torch.long),  # Valid object count.
            "lighting": lighting,  # Lighting condition string.
            "dataset_source": self.dataset_source_key,
        }

        # Add optional fields.
        if audio_tensor is not None:
            result["audio"] = audio_tensor  # [1, 13, T] MFCC features.

        if self.condition_mode:
            result["condition_mode"] = self.condition_mode

        return result
