"""Immutable, versioned split manifest (ticket 309).

A split manifest assigns whole groups (a recording session or source video, ``ImageRecord.group_id``) to
train / val / test, holds out at least one whole environment as ``test_holdout``, and is written once:
a file is never overwritten, and a content hash detects later edits. Changing a split means a new version
with a recorded reason.

Assignment reuses ``ml.datasets.splits.assign_splits``; this module adds the environment rules, the file
format and the leakage check.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from ml.datasets.audit import content_duplicates, has_content_hash
from ml.datasets.manifest import ImageRecord
from ml.datasets.splits import assign_splits

SPLIT_MANIFEST_FORMAT = "maxsight.split_manifest/1"
HOLDOUT_SPLIT = "test_holdout"


class SplitManifestError(ValueError):
    """The records or the manifest break a split rule."""


@dataclass(frozen=True)
class SplitManifest:
    version: str
    reason: str
    seed: int
    ratios: tuple[float, float, float]
    holdout_environments: tuple[str, ...]
    assignments: dict[str, str]  # group_id -> split
    environments: dict[str, str]  # group_id -> environment
    created_at: str = field(default="", compare=False)

    @property
    def content_hash(self) -> str:
        """SHA-256 over everything that defines the split (not the creation time)."""
        body = {
            "version": self.version, "reason": self.reason, "seed": self.seed, "ratios": list(self.ratios),
            "holdout_environments": sorted(self.holdout_environments),
            "assignments": dict(sorted(self.assignments.items())),
            "environments": dict(sorted(self.environments.items())),
        }
        return hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()


def _group_environments(records: list[ImageRecord]) -> dict[str, str]:
    """group_id -> environment; every record needs one, and a group may not span two."""
    environments: dict[str, str] = {}
    for record in records:
        environment = record.extra.get("environment", "")
        if not environment:
            raise SplitManifestError(f"record {record.image_id} has no environment (extra.environment)")
        previous = environments.setdefault(record.group_id, environment)
        if previous != environment:
            raise SplitManifestError(
                f"group {record.group_id} spans more than one environment ({previous}, {environment})")
    return environments


def build_split_manifest(
    records: list[ImageRecord], *, version: str, holdout_environments: tuple[str, ...], reason: str,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15), seed: int = 0,
) -> SplitManifest:
    if not version:
        raise SplitManifestError("a split manifest needs a version")
    if not reason:
        raise SplitManifestError("a split manifest needs a reason")
    if not holdout_environments:
        raise SplitManifestError("hold out at least one environment as test")

    environments = _group_environments(records)
    unknown = sorted(set(holdout_environments) - set(environments.values()))
    if unknown:
        raise SplitManifestError(f"held-out environments not in the data: {unknown}")
    _refuse_content_shared_across_groups(records)

    split_records = assign_splits(records, ratios=ratios, holdout_groups=frozenset(holdout_environments), seed=seed)
    problems = check_leakage(split_records)
    if problems:
        raise SplitManifestError(f"leakage while building the split: {problems[:5]}")

    assignments = {record.group_id: record.split for record in split_records}
    return SplitManifest(
        version=version, reason=reason, seed=seed, ratios=ratios,
        holdout_environments=tuple(sorted(holdout_environments)), assignments=assignments,
        environments=environments, created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


def _missing_content_hashes(records: list[ImageRecord]) -> list[str]:
    return [f"missing_content_hash:{r.image_id}" for r in records if not has_content_hash(r)]


def _refuse_content_shared_across_groups(records: list[ImageRecord]) -> None:
    """Groups are assigned at random, so a shared image would leak on some seeds only: refuse it up front."""
    missing = _missing_content_hashes(records)
    if missing:
        raise SplitManifestError(f"records without sha256 or dhash cannot be checked for leakage: {missing[:5]}")
    shared = [f"content_shared_across_groups:{a.image_id}:{b.image_id}"
              for a, b, _ in content_duplicates(records) if a.group_id != b.group_id]
    if shared:
        raise SplitManifestError(f"the same image is in more than one group; dedupe or merge them first: {shared[:5]}")


def check_leakage(records: list[ImageRecord]) -> list[str]:
    """Problems that would let evaluation data leak into training; empty means none.

    * a record has neither sha256 nor dhash, so its content cannot be cleared;
    * the same image (equal sha256, or dhash within the near-duplicate distance) is in two splits;
    * a group (session/video) appears in more than one split;
    * an environment that has any ``test_holdout`` records also appears in another split.
    """
    problems = _missing_content_hashes(records)
    problems += [f"{kind}_across_splits:{a.image_id}:{b.image_id}"
                 for a, b, kind in content_duplicates(records) if a.split != b.split]
    group_splits: dict[str, set[str]] = {}
    environment_splits: dict[str, set[str]] = {}
    for record in records:
        group_splits.setdefault(record.group_id, set()).add(record.split)
        environment = record.extra.get("environment", "")
        if environment:
            environment_splits.setdefault(environment, set()).add(record.split)

    problems += [f"group_in_multiple_splits:{group}:{sorted(splits)}"
                 for group, splits in sorted(group_splits.items()) if len(splits) > 1]
    problems += [f"environment_split_across_holdout:{env}:{sorted(splits)}"
                 for env, splits in sorted(environment_splits.items())
                 if HOLDOUT_SPLIT in splits and len(splits) > 1]
    return problems


def apply_split_manifest(records: list[ImageRecord], manifest: SplitManifest) -> list[ImageRecord]:
    """Apply assignments only when group identities and environments still match."""
    missing = sorted({r.group_id for r in records} - set(manifest.assignments))
    if missing:
        raise SplitManifestError(f"groups not in the split manifest (new version needed): {missing[:5]}")

    changed_environments = sorted({
        (record.group_id, manifest.environments.get(record.group_id, ""), record.extra.get("environment", ""))
        for record in records
        if manifest.environments.get(record.group_id, "") != record.extra.get("environment", "")
    })
    if changed_environments:
        raise SplitManifestError(
            "group environments differ from the split manifest (new version needed): "
            f"{changed_environments[:5]}"
        )
    return [record.with_(split=manifest.assignments[record.group_id]) for record in records]


def write_split_manifest(manifest: SplitManifest, path: Path | str) -> Path:
    """Write once. An existing file is never replaced: a changed split needs a new version and path."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"split manifest already exists (immutable): {path}")
    doc = {
        "format": SPLIT_MANIFEST_FORMAT, "version": manifest.version, "reason": manifest.reason,
        "created_at": manifest.created_at, "seed": manifest.seed, "ratios": list(manifest.ratios),
        "holdout_environments": list(manifest.holdout_environments),
        "assignments": dict(sorted(manifest.assignments.items())),
        "environments": dict(sorted(manifest.environments.items())),
        "content_hash": manifest.content_hash,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return path


def read_split_manifest(path: Path | str) -> SplitManifest:
    """Read a split manifest and refuse it if its content hash no longer matches (it was edited)."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("format") != SPLIT_MANIFEST_FORMAT:
        raise SplitManifestError(f"unsupported split manifest format: {doc.get('format')}")
    manifest = SplitManifest(
        version=doc["version"], reason=doc["reason"], seed=doc["seed"], ratios=tuple(doc["ratios"]),
        holdout_environments=tuple(doc["holdout_environments"]), assignments=doc["assignments"],
        environments=doc["environments"], created_at=doc.get("created_at", ""),
    )
    if manifest.content_hash != doc.get("content_hash"):
        raise SplitManifestError(f"content hash mismatch: {path} was edited after it was written")
    return manifest
