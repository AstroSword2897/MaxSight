#!/usr/bin/env python3
"""Single-image inference sanity check: load model, run detection, compare to GT with IoU.
Uses MODEL_SIZE (224) for both GT and pred so they are in the same coordinate space.
Works in Colab/notebook (no __file__)."""

import json
import random
import sys
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
IMAGE_DIR = Path("/content/drive/MyDrive/MaxSight_Training")
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
    path = image_dir / rel if not Path(rel).is_absolute() else Path(rel)
    # If path from JSON doesn't exist (e.g. different Drive layout), try under image_dir
    if not path.exists():
        candidates = [image_dir / path.name, image_dir / rel]
        if path.is_absolute():
            for p in ("val2017", "train2017", "images", "val", "train"):
                if p in path.parts:
                    idx = path.parts.index(p)
                    sub = Path(*path.parts[idx:])
                    candidates.append(image_dir / sub)
                    break
        for p in candidates:
            if p.exists():
                path = p
                break
    if not path.exists():
        raise FileNotFoundError(
            f"Image not found: {path}. Tried under IMAGE_DIR={image_dir}. "
            "Set IMAGE_DIR to the folder that contains your images (e.g. where val2017/ lives)."
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

    ann = random.choice([a for a in data if a.get("objects")])

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

    print("GT boxes:", len(gt_boxes))

    model = load_model(CONDITION, CHECKPOINT_DIR, DEVICE)
    image_tensor = load_image_tensor(img_info, IMAGE_DIR, CONDITION, DEVICE)

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
        print("⚠ Boxes don't overlap → scaling/class mismatch.")
    else:
        print("✅ Predictions overlap GT.")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
