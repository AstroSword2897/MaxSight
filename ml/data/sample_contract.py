"""Canonical keys for image-detection training samples.

``MaxSightDataset`` and ``collate_fn`` use these field names so list-format and
COCO-dict adapters converge on one tensor layout before the train loop.
"""

# Primary tensors (single-frame detection).
KEY_IMAGES = "images"
KEY_LABELS = "labels"
KEY_BOXES = "boxes"
KEY_DISTANCE = "distance"
KEY_NUM_OBJECTS = "num_objects"
KEY_URGENCY = "urgency"
KEY_LIGHTING = "lighting"
# Lineage for multi-source runs (registry key ``id@version`` per sample).
KEY_DATASET_SOURCE = "dataset_source"

__all__ = [
    "KEY_IMAGES",
    "KEY_LABELS",
    "KEY_BOXES",
    "KEY_DISTANCE",
    "KEY_NUM_OBJECTS",
    "KEY_URGENCY",
    "KEY_LIGHTING",
    "KEY_DATASET_SOURCE",
]
