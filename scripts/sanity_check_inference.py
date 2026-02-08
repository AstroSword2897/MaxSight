#!/usr/bin/env python3
# Loads the model, runs detection on one image, and compares predictions to ground truth with IoU. Uses 224 for both. Works in Colab where __file__ is not set.

import json
import os
import random
import sys
from collections import deque
from pathlib import Path

import torch
from PIL import Image

try:
    REPO_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    REPO_ROOT = Path.cwd()
sys.path.insert(0, str(REPO_ROOT))

from ml.models.maxsight_cnn import (
    COCO_CLASSES,
    CapabilityTier,
    TierConfig,
    create_model,
)
from ml.utils.preprocessing import ImagePreprocessor

VAL_JSON = "/content/drive/MyDrive/MaxSight_Training/cleaned_splits/maxsight_val.json"

IMAGE_DIR = Path(os.environ.get("IMAGE_DIR", "/content/drive/MyDrive/MaxSight_Training"))
CHECKPOINT_DIR = Path("/content/drive/MyDrive/MaxSight")
CONDITION = "cvi"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_SIZE = 224


def load_model(condition, checkpoint_dir, device):
    model = create_model(
        num_classes=len(COCO_CLASSES),
        use_audio=False,
        condition_mode=condition,
        tier_config=TierConfig.for_tier(CapabilityTier["T5_TEMPORAL"]),
    )
    ckpt_path = checkpoint_dir / f"checkpoints_{condition}" / "best_model.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def load_image_tensor(img_info, image_dir, condition, device):
    rel = img_info.get("file_name", img_info.get("image_path"))
    filename = Path(rel).name

    # Annotation paths may be stale; the filename is looked up in common folder layouts.
    candidates = [
        image_dir / filename,
        image_dir / "val2017" / filename,
        image_dir / "train2017" / filename,
        image_dir / "images" / filename,
        image_dir / "datasets" / "coco_raw" / "val2017" / filename,
        image_dir / "datasets" / "coco_raw" / "train2017" / filename,
        image_dir / "coco_raw" / "val2017" / filename,
        image_dir / "coco_raw" / "train2017" / filename,
    ]
    if not Path(rel).is_absolute():
        candidates.append(image_dir / rel)
    path = None
    for p in candidates:
        if p.exists():
            path = p
            break
    if path is None and image_dir.exists():
        # Search the directory tree so val/train folders are checked before stopping.
        max_depth = 8
        max_dirs = 5000
        try:
            queue = deque([(image_dir, 0)])
            dirs_done = 0
            while queue and dirs_done < max_dirs:
                root, depth = queue.popleft()
                if depth > max_depth:
                    continue
                dirs_done += 1
                try:
                    entries = list(Path(root).iterdir())
                except OSError:
                    continue
                for e in entries:
                    if e.is_file():
                        if e.name == filename:
                            path = e
                            queue.clear()
                            break
                    elif e.is_dir():
                        queue.append((e, depth + 1))
                if path is not None:
                    break
        except (ValueError, OSError):
            pass
    if path is None or not path.exists():
        raise FileNotFoundError(
            f"Image not found for {filename}. Tried fixed paths and search under IMAGE_DIR={image_dir}. "
            "Ensure the image exists under that folder (or set os.environ['IMAGE_DIR'] to the correct base)."
        )
    pil = Image.open(path).convert("RGB")
    preprocessor = ImagePreprocessor(
        image_size=(MODEL_SIZE, MODEL_SIZE),
        condition_mode=condition,
    )
    tensor = preprocessor(pil)
    return tensor.unsqueeze(0).to(device)


def get_detections(model, images):
    with torch.no_grad():
        outputs = model(images)
    return model.get_detections(outputs, confidence_threshold=0.001)


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter)


def main():
    with open(VAL_JSON) as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("Expected MaxSight format: JSON list of annotations with 'objects' and 'image_path'.")
        return 1

    candidates = [a for a in data if a.get("objects")]
    if not candidates:
        print("No annotations with objects in JSON.")
        return 1
    random.shuffle(candidates)

    ann = None
    gt_boxes = []
    gt_classes = []
    img_info = None
    last_err = None
    max_tries = min(15, len(candidates))

    for i in range(max_tries):
        ann = candidates[i]
        img_info = {
            "file_name": ann["image_path"],
            "image_path": ann["image_path"],
        }
        gt_boxes = []
        gt_classes = []
        for obj in ann["objects"]:
            cx, cy, w, h = obj["box"]
            x1 = (cx - w / 2) * MODEL_SIZE
            y1 = (cy - h / 2) * MODEL_SIZE
            x2 = (cx + w / 2) * MODEL_SIZE
            y2 = (cy + h / 2) * MODEL_SIZE
            gt_boxes.append([x1, y1, x2, y2])
            gt_classes.append(obj.get("class", 0))
        try:
            image_tensor = load_image_tensor(img_info, IMAGE_DIR, CONDITION, DEVICE)
            break
        except FileNotFoundError as e:
            last_err = e
            if i < max_tries - 1:
                continue
            raise last_err from None

    print("GT boxes:", len(gt_boxes))

    model = load_model(CONDITION, CHECKPOINT_DIR, DEVICE)

    detections = get_detections(model, image_tensor)[0]
    print("Predicted boxes:", len(detections))

    if len(detections) == 0:
        print("Model predicts nothing.")
        return 1

    pred_boxes = []
    pred_classes = []

    for d in detections:
        cx, cy, w, h = d["box"]
        x1 = (cx - w / 2) * MODEL_SIZE
        y1 = (cy - h / 2) * MODEL_SIZE
        x2 = (cx + w / 2) * MODEL_SIZE
        y2 = (cy + h / 2) * MODEL_SIZE
        pred_boxes.append([x1, y1, x2, y2])
        pred_classes.append(d["class"])

    print("Pred classes:", set(pred_classes))
    print("GT classes:", set(gt_classes))

    max_iou_val = 0
    for pb in pred_boxes:
        for gb in gt_boxes:
            max_iou_val = max(max_iou_val, iou(pb, gb))

    print("Max IoU:", max_iou_val)

    if max_iou_val < 0.1:
        print("WARNING Boxes don't overlap - scaling/class mismatch.")
    else:
        print("OK Predictions overlap GT.")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())





