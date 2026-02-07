#!/usr/bin/env python3
"""Single-image inference sanity check: load model, run detection, compare to GT with IoU. Works in Colab/notebook (no __file__)."""

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

# ---- CONFIG ----
VAL_JSON = "/content/drive/MyDrive/MaxSight_Training/cleaned_splits/maxsight_val.json"
IMAGE_DIR = Path("/content/drive/MyDrive/MaxSight_Training")
CHECKPOINT_DIR = Path("/content/drive/MyDrive/MaxSight")
CONDITION = "cvi"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------


def load_model(condition: str, checkpoint_dir: Path, device: str):
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


def load_image_tensor(img_info: dict, image_dir: Path, condition: str, device: str) -> torch.Tensor:
    # Support both "file_name" (COCO) and "image_path" (MaxSight list format)
    rel = img_info.get("file_name", img_info.get("image_path", "image.jpg"))
    path = Path(rel) if Path(rel).is_absolute() else image_dir / rel
    pil = Image.open(path).convert("RGB")
    preprocessor = ImagePreprocessor(image_size=(224, 224), condition_mode=condition)
    tensor = preprocessor(pil)
    return tensor.unsqueeze(0).to(device)


def get_detections(model, images: torch.Tensor, confidence_threshold: float = 0.001):
    with torch.no_grad():
        outputs = model(images)
    return model.get_detections(outputs, confidence_threshold=confidence_threshold)


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

    # Support COCO format (dict with "images"/"annotations") or MaxSight format (list of anns)
    if isinstance(data, dict) and "images" in data and "annotations" in data:
        img_info = random.choice(data["images"])
        image_id = img_info["id"]
        img_w = img_info.get("width", 224)
        img_h = img_info.get("height", 224)
        gt_annos = [a for a in data["annotations"] if a["image_id"] == image_id]
        sx, sy = 224 / max(1, img_w), 224 / max(1, img_h)
        gt_boxes = []
        gt_classes = []
        for ann in gt_annos:
            x, y, w, h = ann["bbox"]
            x1, y1 = x * sx, y * sy
            x2, y2 = (x + w) * sx, (y + h) * sy
            gt_boxes.append([x1, y1, x2, y2])
            gt_classes.append(ann["category_id"])
    elif isinstance(data, list):
        # MaxSight cleaned_splits: list of { image_id, image_path, objects: [{ box: [cx,cy,w,h], class }] }
        candidates = [a for a in data if a.get("objects")]
        if not candidates:
            print("No annotations with objects in JSON.")
            return 1
        ann = random.choice(candidates)
        image_id = ann.get("image_id", ann.get("id", 0))
        img_info = {
            "file_name": ann.get("image_path", f"{image_id}.jpg"),
            "image_path": ann.get("image_path", f"{image_id}.jpg"),
        }
        # GT boxes already normalized [cx, cy, w, h] -> convert to 224 xyxy
        gt_boxes = []
        gt_classes = []
        for obj in ann["objects"]:
            cx, cy, w, h = obj["box"]
            x1 = (cx - w / 2) * 224
            y1 = (cy - h / 2) * 224
            x2 = (cx + w / 2) * 224
            y2 = (cy + h / 2) * 224
            gt_boxes.append([x1, y1, x2, y2])
            gt_classes.append(obj.get("class", 0))
    else:
        print("Unknown JSON format: need 'images'/'annotations' (COCO) or list of annotations (MaxSight).")
        return 1

    print("GT boxes:", len(gt_boxes))

    if len(gt_boxes) == 0:
        print("No GT boxes — pick another image or check dataset.")
        return 1

    image_tensor = load_image_tensor(img_info, IMAGE_DIR, CONDITION, DEVICE)
    model = load_model(CONDITION, CHECKPOINT_DIR, DEVICE)
    detections = get_detections(model, image_tensor, confidence_threshold=0.001)
    print("Predicted boxes:", len(detections[0]))

    if len(detections[0]) == 0:
        print("Model predicts nothing.")
        return 1

    pred_boxes = []
    pred_classes = []
    for d in detections[0]:
        cx, cy, w, h = d["box"]
        x1 = (cx - w / 2) * 224
        y1 = (cy - h / 2) * 224
        x2 = (cx + w / 2) * 224
        y2 = (cy + h / 2) * 224
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
        print("Boxes don't overlap — scaling/format issue.")
    else:
        print("Predictions overlap GT.")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
