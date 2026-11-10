"""Ground-truth guard for safety gates (ticket 312).

SG-01 and SG-02 measure real performance, so their ground truth must be real and human-reviewed. A
record is rejected if it is flagged synthetic, or if its annotation_source is anything other than
"human_reviewed" (synthetic, heuristic, model predictions, unreviewed MaxSight labels and public
dataset labels are all rejected). One bad record fails the whole run: nothing is silently filtered.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from ml.datasets.manifest import ImageRecord, read_manifest

ACCEPTED_SOURCE = "human_reviewed"
MAX_REASONS_SHOWN = 10


class GroundTruthRejected(ValueError):
    """Raised when safety-gate ground truth contains anything that is not human-reviewed real data."""


def rejection_reason(record: ImageRecord) -> str:
    """Why ``record`` cannot be safety-gate ground truth, or "" if it can."""
    if record.is_synthetic or record.annotation_source == "synthetic":
        return "synthetic"
    if record.annotation_source != ACCEPTED_SOURCE:
        return f"not human-reviewed (annotation_source={record.annotation_source or 'missing'})"
    return ""


def assert_human_reviewed(records: Iterable[ImageRecord]) -> int:
    """Return the number of records, or raise GroundTruthRejected if any record is not acceptable."""
    count = 0
    rejected: list[str] = []
    for record in records:
        count += 1
        reason = rejection_reason(record)
        if reason:
            rejected.append(f"{record.image_id}: {reason}")

    if count == 0:
        raise GroundTruthRejected("no records: an empty manifest is not ground truth")
    if rejected:
        shown = "; ".join(rejected[:MAX_REASONS_SHOWN])
        more = f" (+{len(rejected) - MAX_REASONS_SHOWN} more)" if len(rejected) > MAX_REASONS_SHOWN else ""
        raise GroundTruthRejected(f"{len(rejected)} of {count} records rejected: {shown}{more}")
    return count


def load_ground_truth(manifest: Path | str) -> list[ImageRecord]:
    """Read a ground-truth manifest and apply the guard."""
    records = list(read_manifest(manifest))
    assert_human_reviewed(records)
    return records
