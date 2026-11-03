"""Public-source registry, access states and usage policy.

State is derived from marker files on disk, never from configuration, so a dataset that is
only configured can never be reported as downloaded.
"""

from __future__ import annotations

import json
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
    CONFIGURED = 0
    DOWNLOADED = 1  # files present, integrity not yet verified
    VERIFIED = 2
    PROCESSED = 3
    VALIDATED = 4


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
        return State.PROCESSED
    marker = data_root / "raw" / src.id / ACQUIRED_MARKER
    if marker.exists():
        info = json.loads(marker.read_text(encoding="utf-8"))
        return State.VERIFIED if info.get("verified") else State.DOWNLOADED
    return State.CONFIGURED


def assert_detector_training_sources(source_ids: set[str], registry: dict[str, Source]) -> None:
    """Only ``pretrain`` sources may enter the detector mix (BDD100K is eval-only; research sets are separate)."""
    bad = {s: registry[s].role for s in source_ids if s in registry and registry[s].role != "pretrain"}
    unknown = sorted(s for s in source_ids if s not in registry and not s.startswith("maxsight"))
    if bad or unknown:
        raise ValueError(f"not allowed in detector training: roles={bad} unregistered={unknown}")


def commercial_release_blockers(source_ids: set[str], registry: dict[str, Source]) -> list[str]:
    return sorted(f"{s}:{registry[s].license_id}" for s in source_ids if s in registry and registry[s].non_commercial)
