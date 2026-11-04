"""Open Images V7 subset via FiftyOne (optional dependency; downloads only requested classes)."""

from __future__ import annotations

from pathlib import Path

from ml.datasets.label_map import LabelMap
from ml.datasets.manifest import BoxRecord, ImageRecord

SOURCE = "openimages"
LICENSE = "annotations CC BY 4.0; images CC BY 2.0"


def fetch(label_map: LabelMap, root: Path, max_per_class: int = 4000, split: str = "train"):
    """Download images for every mapped Open Images class. Requires ``pip install fiftyone``."""
    try:
        import fiftyone.zoo as foz
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit("Open Images needs FiftyOne: pip install fiftyone") from exc
    classes = sorted(label_map.sources[SOURCE])
    return foz.load_zoo_dataset(
        "open-images-v7", split=split, label_types=["detections"], classes=classes,
        max_samples=max_per_class * len(classes), dataset_dir=str(root),
    )


def to_records(dataset, label_map: LabelMap) -> list[ImageRecord]:  # pragma: no cover - needs FiftyOne data
    out: list[ImageRecord] = []
    for sample in dataset:
        w, h = sample.metadata.width, sample.metadata.height
        boxes: list[BoxRecord] = []
        for det in (sample.ground_truth.detections if sample.ground_truth else []):
            target = label_map.map(SOURCE, det.label)
            if target is None:
                continue
            x, y, bw, bh = det.bounding_box  # relative xywh
            boxes.append(BoxRecord(target, (x * w, y * h, (x + bw) * w, (y + bh) * h), det.label))
        if boxes:
            iid = Path(sample.filepath).stem
            out.append(ImageRecord(f"openimages:{iid}", SOURCE, sample.filepath, w, h, f"openimages:{iid}",
                                   tuple(boxes), license=LICENSE))
    return out
