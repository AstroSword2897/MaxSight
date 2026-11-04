"""Image manifest records (JSONL): one line per image, boxes in source pixels."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

MANIFEST_VERSION = "maxsight.manifest/1"


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
    if d.pop("manifest_version", None) != MANIFEST_VERSION:
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
