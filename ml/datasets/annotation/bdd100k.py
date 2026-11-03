"""BDD100K detection labels -> evaluation manifest. Evaluation only: records are tagged ``split=eval``.

Input is the official detection label list (``[{name, attributes{weather,scene,timeofday}, labels[{category, box2d}]}]``).
The original JSON is never modified. Camera-domain attributes are kept in ``extra`` for stratified reporting.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from ml.datasets.label_map import LabelMap
from ml.datasets.manifest import BoxRecord, ImageRecord

SOURCE = "bdd100k"
LICENSE = "BDD100K research licence (evaluation only)"
DEFAULT_SIZE = (1280, 720)  # BDD100K frames are 1280x720; verified from the file when it exists


def to_records(labels_path: Path, label_map: LabelMap, image_root: Path | None = None,
               image_subdir: str = "images/100k/val") -> list[ImageRecord]:
    items = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    out: list[ImageRecord] = []
    for item in items:
        name = item["name"]
        w, h = DEFAULT_SIZE
        rel = f"{image_subdir}/{name}"
        if image_root is not None and (image_root / rel).exists():
            with Image.open(image_root / rel) as im:
                w, h = im.size
        boxes: list[BoxRecord] = []
        ignores: list[tuple[float, float, float, float]] = []
        for lab in item.get("labels", []):
            b = lab.get("box2d")
            if not b:
                continue  # polygon/lane labels are not detection boxes
            xyxy = (float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"]))
            cat = lab["category"]
            target = label_map.map(SOURCE, cat)
            if target is not None:
                boxes.append(BoxRecord(target, xyxy, cat))
            elif label_map.is_ignored(SOURCE, cat):
                ignores.append(xyxy)
        attrs = item.get("attributes", {}) or {}
        out.append(ImageRecord(
            image_id=f"{SOURCE}:{name}", source=SOURCE, path=rel, width=w, height=h,
            group_id=f"{SOURCE}:{name.split('-')[0]}",  # id prefix groups frames of one drive
            boxes=tuple(boxes), ignore_regions=tuple(ignores), split="eval", license=LICENSE,
            extra={"weather": str(attrs.get("weather", "")), "scene": str(attrs.get("scene", "")),
                   "timeofday": str(attrs.get("timeofday", "")), "camera_domain": "vehicle_mounted"},
        ))
    return out
