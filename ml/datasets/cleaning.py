"""Dataset cleaning: decode check, size filter, perceptual-hash dedup, per-class capping."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from ml.datasets.manifest import ImageRecord

HASH_BITS = 64
_CHUNKS = 5  # pigeonhole: Hamming <= 4 over 5 chunks leaves one chunk identical
NEAR_DUP_MAX_HAMMING = _CHUNKS - 1  # largest threshold the chunk index can honour exactly


def dhash(img: Image.Image) -> int:
    """64-bit difference hash."""
    g = img.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    px = list(g.getdata())
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | (px[row * 9 + col] > px[row * 9 + col + 1])
    return bits


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _chunk_keys(h: int) -> list[tuple[int, int]]:
    size = -(-HASH_BITS // _CHUNKS)
    return [(i, (h >> (i * size)) & ((1 << size) - 1)) for i in range(_CHUNKS)]


@dataclass
class CleanReport:
    kept: int = 0
    dropped: Counter = field(default_factory=Counter)
    boxes_dropped_small: int = 0

    def as_dict(self) -> dict:
        return {"kept": self.kept, "dropped": dict(self.dropped), "boxes_dropped_small": self.boxes_dropped_small}


def clean(
    records: list[ImageRecord],
    root: Path,
    min_box_px: float = 8.0,
    dup_hamming: int = 4,
    per_class_cap: int | None = None,
    verify_files: bool = True,
) -> tuple[list[ImageRecord], CleanReport]:
    """Return cleaned records and a report. Records keep a fixed order (input order)."""
    if not 0 <= dup_hamming <= NEAR_DUP_MAX_HAMMING:
        raise ValueError(f"dup_hamming must be 0..{NEAR_DUP_MAX_HAMMING}: the chunk index misses larger distances")
    report = CleanReport()
    index: dict[tuple[int, int], list[int]] = defaultdict(list)
    seen_hashes: list[int] = []
    kept: list[ImageRecord] = []

    for rec in records:
        boxes = []
        ignores = list(rec.ignore_regions)
        for b in rec.boxes:
            x1, y1, x2, y2 = b.bbox_xyxy
            if min(x2 - x1, y2 - y1) < min_box_px or x2 <= x1 or y2 <= y1:
                report.boxes_dropped_small += 1
                if x2 > x1 and y2 > y1:  # a real but tiny object: ignore it, never leave it as background
                    ignores.append(b.bbox_xyxy)
                continue
            boxes.append(b)
        if not boxes:
            report.dropped["no_boxes_after_filter"] += 1
            continue

        digest, h = rec.sha256, int(rec.dhash, 16) if rec.dhash else None
        if verify_files:
            path = root / rec.path
            try:
                data = path.read_bytes()
                with Image.open(path) as im:
                    im.verify()
                with Image.open(path) as im:
                    w, ht = im.size
                    h = dhash(im)
            except Exception:  # noqa: BLE001 - any decode failure drops the image
                report.dropped["corrupt_or_missing"] += 1
                continue
            digest = hashlib.sha256(data).hexdigest()
            if (w, ht) != (rec.width, rec.height):
                rec = rec.with_(width=w, height=ht)
        if h is not None:
            dup = False
            for key in _chunk_keys(h):
                if any(hamming(h, seen_hashes[j]) <= dup_hamming for j in index[key]):
                    dup = True
                    break
            if dup:
                report.dropped["near_duplicate"] += 1
                continue
            for key in _chunk_keys(h):
                index[key].append(len(seen_hashes))
            seen_hashes.append(h)
        kept.append(rec.with_(boxes=tuple(boxes), ignore_regions=tuple(ignores), sha256=digest, dhash=f"{h:016x}" if h is not None else ""))

    if per_class_cap is not None:
        kept = cap_per_class(kept, per_class_cap, report)
    report.kept = len(kept)
    return kept, report


def cap_per_class(records: list[ImageRecord], cap: int, report: CleanReport | None = None) -> list[ImageRecord]:
    """Drop whole images (never single boxes) whose every class is already at the cap."""
    counts: Counter[str] = Counter()
    out: list[ImageRecord] = []
    for rec in records:
        if all(counts[b.cls] >= cap for b in rec.boxes):
            if report is not None:
                report.dropped["class_cap"] += 1
            continue
        counts.update(b.cls for b in rec.boxes)
        out.append(rec)
    return out
