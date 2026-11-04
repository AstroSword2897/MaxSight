"""Public-source registry, access states and usage policy.

State is derived from marker files on disk, never from configuration, so a dataset that is
only configured can never be reported as downloaded.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = REPO / "configs" / "data" / "sources.yaml"
DEFAULT_DATA_ROOT = REPO / "datasets"

ROLES = ("pretrain", "eval_only", "research")
ACCESS_KINDS = ("automatic", "registration", "license_acceptance", "credentials", "manual_download")
ACQUIRED_MARKER = "ACQUIRED.json"  # written under raw/<id>/ after files are present and checksummed
VALIDATED_MARKER = "VALIDATED.json"  # written under processed/<id>/ after the audit passes


class State(IntEnum):
    """How far a dataset has been checked. Each step needs evidence on disk, never configuration."""

    NOT_DOWNLOADED = 0  # registered only
    DOWNLOADED_UNVERIFIED = 1  # files present, integrity not yet checked
    RAW_VERIFIED = 2  # every split passed the integrity check
    PREPROCESSED_UNVERIFIED = 3  # processed output exists, quality audit not passed yet
    VALIDATED = 4  # quality audit passed; see status_label() for what it is ready for


# A validated dataset is ready only for the purpose its registry role allows.
READY_LABEL_BY_ROLE = {
    "pretrain": "READY_FOR_DETECTOR_PRETRAINING",
    "eval_only": "READY_FOR_EVALUATION_ONLY",
    "research": "READY_FOR_RESEARCH_EXPERIMENTS",
}


@dataclass(frozen=True)
class Source:
    id: str
    name: str
    role: str
    version: str
    homepage: str
    license_id: str
    commercial_use: str  # "true" | "false" | "unverified"
    access_kind: str
    access_note: str
    raw: dict

    @property
    def non_commercial(self) -> bool:
        return self.commercial_use != "true"


def load_registry(path: Path | str = DEFAULT_REGISTRY) -> dict[str, Source]:
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    out: dict[str, Source] = {}
    for sid, d in doc["sources"].items():
        if d["role"] not in ROLES:
            raise ValueError(f"{sid}: unknown role {d['role']!r}")
        if d["access"]["kind"] not in ACCESS_KINDS:
            raise ValueError(f"{sid}: unknown access kind {d['access']['kind']!r}")
        lic = d["license"]
        out[sid] = Source(
            id=sid, name=d["name"], role=d["role"], version=str(d["version"]), homepage=d["homepage"],
            license_id=lic["id"], commercial_use=str(lic["commercial_use"]).lower(),
            access_kind=d["access"]["kind"], access_note=d["access"].get("note", ""), raw=d,
        )
    return out


def current_state(src: Source, data_root: Path = DEFAULT_DATA_ROOT) -> State:
    """Highest state with on-disk evidence."""
    validated = data_root / "processed" / src.id / VALIDATED_MARKER
    if validated.exists():
        return State.VALIDATED
    if (data_root / "processed" / src.id).is_dir() and any((data_root / "processed" / src.id).iterdir()):
        return State.PREPROCESSED_UNVERIFIED
    marker = data_root / "raw" / src.id / ACQUIRED_MARKER
    if marker.exists():
        info = json.loads(marker.read_text(encoding="utf-8"))
        return State.RAW_VERIFIED if info.get("verified") else State.DOWNLOADED_UNVERIFIED
    return State.NOT_DOWNLOADED


def status_label(src: Source, data_root: Path = DEFAULT_DATA_ROOT) -> str:
    """The state name, except that VALIDATED is reported as the purpose it is ready for."""
    state = current_state(src, data_root)
    if state is State.VALIDATED:
        return READY_LABEL_BY_ROLE[src.role]
    return state.name


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_marker_splits(marker: Path) -> dict[str, dict]:
    """Existing per-split entries. A legacy marker (one top-level ``split``) becomes one entry."""
    if not marker.exists():
        return {}
    old = json.loads(marker.read_text(encoding="utf-8"))
    if "splits" in old:
        return dict(old["splits"])
    if "split" in old:
        legacy = {key: value for key, value in old.items() if key != "split"}
        return {old["split"]: legacy}
    return {}


def write_acquired_marker(raw: Path, *, verified: bool, split: str | None = None, **info) -> Path:
    """Record that files are on disk.

    ``verified`` is True only after an integrity check has passed. With ``split``, the entry is
    added under ``splits`` and the other splits are kept; the top-level ``verified`` is True only
    when every split is verified. Without ``split`` (single-archive sources such as DIODE), the
    marker describes the whole dataset.
    """
    raw.mkdir(parents=True, exist_ok=True)
    marker = raw / ACQUIRED_MARKER

    if split is None:
        payload = {"verified": verified, "acquired_at": _utc_now(), **info}
    else:
        splits = _read_marker_splits(marker)
        splits[split] = {"verified": verified, "acquired_at": _utc_now(), **info}
        payload = {
            "verified": all(entry.get("verified") for entry in splits.values()),
            "updated_at": _utc_now(),
            "splits": splits,
        }

    marker.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return marker


def write_validated_marker(data_root: Path, source_id: str, summary: dict) -> Path:
    """Record a passing audit. Callers must clear it when an audit fails (see clear_validated_marker)."""
    marker = data_root / "processed" / source_id / VALIDATED_MARKER
    marker.parent.mkdir(parents=True, exist_ok=True)
    payload = {**summary, "validated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    marker.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return marker


def clear_validated_marker(data_root: Path, source_id: str) -> bool:
    """Remove a stale VALIDATED marker so a failed re-audit can never keep an old pass."""
    marker = data_root / "processed" / source_id / VALIDATED_MARKER
    if marker.exists():
        marker.unlink()
        return True
    return False


def assert_detector_training_sources(source_ids: set[str], registry: dict[str, Source]) -> None:
    """Only ``pretrain`` sources may enter the detector mix (BDD100K is eval-only; research sets are separate)."""
    bad = {s: registry[s].role for s in source_ids if s in registry and registry[s].role != "pretrain"}
    unknown = sorted(s for s in source_ids if s not in registry and not s.startswith("maxsight"))
    if bad or unknown:
        raise ValueError(f"not allowed in detector training: roles={bad} unregistered={unknown}")


def commercial_release_blockers(source_ids: set[str], registry: dict[str, Source]) -> list[str]:
    return sorted(f"{s}:{registry[s].license_id}" for s in source_ids if s in registry and registry[s].non_commercial)
