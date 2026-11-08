"""Dataset audit: the checks behind `scripts/validate_data.py` (pre-training quality gate)."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from ml.datasets.cleaning import NEAR_DUP_MAX_HAMMING, _chunk_keys, hamming
from ml.datasets.label_map import LabelMap
from ml.datasets.manifest import ImageRecord

MAX_CLASS_SHARE = 0.15


@dataclass
class AuditResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    class_counts: Counter = field(default_factory=Counter)
    split_counts: Counter = field(default_factory=Counter)

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict:
        return {"ok": self.ok, "errors": self.errors, "warnings": self.warnings,
                "class_counts": dict(self.class_counts), "split_counts": dict(self.split_counts)}


def audit(
    records: list[ImageRecord],
    label_map: LabelMap,
    root: Path | None = None,
    max_class_share: float = MAX_CLASS_SHARE,
    min_instances_per_class: int = 1,
) -> AuditResult:
    res = AuditResult()
    ids = Counter(r.image_id for r in records)
    res.errors += [f"duplicate_image_id:{i}" for i, n in ids.items() if n > 1]

    for r in records:
        res.split_counts[r.split or "unassigned"] += 1
        for b in r.boxes:
            if b.cls not in label_map.classes:
                res.errors.append(f"unknown_class:{r.image_id}:{b.cls}")
            res.class_counts[b.cls] += 1
            x1, y1, x2, y2 = b.bbox_xyxy
            if not (0 <= x1 < x2 <= r.width + 1 and 0 <= y1 < y2 <= r.height + 1):
                res.errors.append(f"box_out_of_bounds:{r.image_id}")
        if root is not None and not (root / r.path).exists():
            res.errors.append(f"missing_file:{r.image_id}")

    total = sum(res.class_counts.values())
    for cls in label_map.classes:
        n = res.class_counts.get(cls, 0)
        if n < min_instances_per_class:
            res.warnings.append(f"class_under_minimum:{cls}:{n}")
        if total and n / total > max_class_share:
            res.errors.append(f"class_share_exceeds_{max_class_share:.2f}:{cls}:{n / total:.3f}")

    groups: dict[str, set[str]] = defaultdict(set)
    for r in records:
        if r.split:
            groups[r.group_id].add(r.split)
    res.errors += [f"group_leak:{g}:{sorted(s)}" for g, s in groups.items() if len(s) > 1]

    in_splits = [r for r in records if r.split]
    unverified = sum(1 for r in in_splits if not has_content_hash(r))
    if unverified:
        res.warnings.append(f"content_unverified:{unverified}_split_records_without_sha256_or_dhash")
    res.errors += [f"{kind}_across_splits:{a.image_id}:{b.image_id}"
                   for a, b, kind in content_duplicates(in_splits) if a.split != b.split]
    return res


def has_content_hash(record: ImageRecord) -> bool:
    return bool(record.sha256 or record.dhash)


def content_duplicates(records: list[ImageRecord]) -> list[tuple[ImageRecord, ImageRecord, str]]:
    """Pairs ``(earlier, later, kind)`` of records holding the same image.

    ``same_content``: equal sha256. Each record is paired with the first record of its hash, which is
    enough to expose any pair that differs in split or group (one of the two must differ from the first).
    ``near_duplicate``: dhash within NEAR_DUP_MAX_HAMMING, found with the chunk index (same pigeonhole
    trick as cleaning), exact at any dataset size so the check is never skipped. It used to be skipped
    above 3000 images while the audit still passed.
    """
    pairs: list[tuple[ImageRecord, ImageRecord, str]] = []
    first_by_sha: dict[str, ImageRecord] = {}
    for r in records:
        if not r.sha256:
            continue
        first = first_by_sha.setdefault(r.sha256, r)
        if first is not r:
            pairs.append((first, r, "same_content"))

    hashed = [(r, int(r.dhash, 16)) for r in records if r.dhash]
    index: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, (a, ha) in enumerate(hashed):
        candidates = {j for key in _chunk_keys(ha) for j in index[key]}
        for j in sorted(candidates):
            b, hb = hashed[j]
            exact_already = bool(a.sha256) and a.sha256 == b.sha256
            if not exact_already and hamming(ha, hb) <= NEAR_DUP_MAX_HAMMING:
                pairs.append((b, a, "near_duplicate"))
        for key in _chunk_keys(ha):
            index[key].append(i)
    return pairs
