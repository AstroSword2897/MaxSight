"""Image manifest records (JSONL): one line per image, boxes in source pixels."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

MANIFEST_VERSION = "maxsight.manifest/2"
READABLE_VERSIONS = ("maxsight.manifest/1", MANIFEST_VERSION)  # v1 loads; it then fails validation

# Where a sample's annotations came from. Only "human_reviewed" may feed safety gates (ticket 312).
ANNOTATION_SOURCES = (
    "source_dataset",  # the public dataset's own annotations
    "human_reviewed",  # MaxSight annotation, reviewed by a second person
    "human_unreviewed",  # MaxSight annotation, not reviewed yet
    "model_prediction",  # produced by a model; never ground truth
    "heuristic",  # produced by rules; never ground truth
    "synthetic",  # generated scene
)
REQUIRED_FIELDS = (
    "source", "dataset_version", "licence_id", "path", "sha256", "label_map_version", "annotation_source",
)


@dataclass(frozen=True)
class BoxRecord:
    cls: str
    bbox_xyxy: tuple[float, float, float, float]
    source_class: str = ""


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    source: str
    path: str
    width: int
    height: int
    group_id: str
    boxes: tuple[BoxRecord, ...] = ()
    ignore_regions: tuple[tuple[float, float, float, float], ...] = ()
    url: str = ""
    sha256: str = ""
    dhash: str = ""
    split: str = ""
    is_synthetic: bool = False
    license: str = ""
    dataset_version: str = ""
    licence_id: str = ""  # short id of the licence that governs this sample (see the licence register)
    label_map_version: str = ""
    annotation_source: str = ""  # one of ANNOTATION_SOURCES
    extra: dict[str, str] = field(default_factory=dict)

    def with_(self, **changes) -> ImageRecord:
        return replace(self, **changes)


def _to_json(rec: ImageRecord) -> str:
    d = asdict(rec)
    d["manifest_version"] = MANIFEST_VERSION
    d["boxes"] = [{"cls": b.cls, "bbox_xyxy": list(b.bbox_xyxy), "source_class": b.source_class} for b in rec.boxes]
    d["ignore_regions"] = [list(r) for r in rec.ignore_regions]
    return json.dumps(d, separators=(",", ":"))


def _from_json(line: str) -> ImageRecord:
    d = json.loads(line)
    if d.pop("manifest_version", None) not in READABLE_VERSIONS:
        raise ValueError("unsupported manifest version")
    d["boxes"] = tuple(BoxRecord(b["cls"], tuple(b["bbox_xyxy"]), b.get("source_class", "")) for b in d["boxes"])
    d["ignore_regions"] = tuple(tuple(r) for r in d["ignore_regions"])
    return ImageRecord(**d)


def write_manifest(records: Iterable[ImageRecord], path: Path | str) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(_to_json(rec) + "\n")
            n += 1
    return n


def read_manifest(path: Path | str) -> Iterator[ImageRecord]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield _from_json(line)


def validate_records(records: Iterable[ImageRecord], *, require_split: bool = False) -> list[str]:
    """Schema errors for ``records``; an empty list means every record is complete.

    A record fails when any REQUIRED_FIELDS value is empty (so a missing licence_id or checksum always
    fails), when its annotation_source is unknown, or when it is flagged synthetic without saying so.
    """
    errors: list[str] = []
    for record in records:
        for name in REQUIRED_FIELDS:
            if not getattr(record, name):
                errors.append(f"missing_{name}:{record.image_id}")
        source = record.annotation_source
        if source and source not in ANNOTATION_SOURCES:
            errors.append(f"unknown_annotation_source:{record.image_id}:{source}")
        if record.is_synthetic and source != "synthetic":
            errors.append(f"synthetic_flag_mismatch:{record.image_id}")
        if require_split and not record.split:
            errors.append(f"missing_split:{record.image_id}")
    return errors
