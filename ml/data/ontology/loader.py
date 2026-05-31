"""Load and validate the seven-disability ontology artifact."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ONTOLOGY_PATH = Path(__file__).resolve().parent / "disability_ontology.json"


@dataclass(frozen=True)
class DisabilityClass:
    """One disability entry in the ontology graph."""

    id: str
    name: str
    category: str
    feature_tags: list[str]
    therapy_focus: list[str]
    model_condition_key: str


class DisabilityOntology:
    """Versioned disability ontology for data labeling and therapy routing."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.version = str(payload.get("version", "0"))
        self._by_id: dict[str, DisabilityClass] = {}
        for item in payload.get("disabilities", []):
            dc = DisabilityClass(
                id=item["id"],
                name=item["name"],
                category=item["category"],
                feature_tags=list(item.get("feature_tags", [])),
                therapy_focus=list(item.get("therapy_focus", [])),
                model_condition_key=item["model_condition_key"],
            )
            self._by_id[dc.id] = dc
        self.edges = list(payload.get("edges", []))

    @classmethod
    def load(cls, path: Path | None = None) -> DisabilityOntology:
        """Load ontology JSON from disk."""
        p = path or ONTOLOGY_PATH
        payload = json.loads(p.read_text(encoding="utf-8"))
        return cls(payload)

    def validate(self) -> None:
        """Raise ValueError when ontology violates production constraints."""
        if len(self._by_id) != 7:
            raise ValueError(f"ontology must define exactly 7 disabilities, got {len(self._by_id)}")
        keys = {dc.model_condition_key for dc in self._by_id.values()}
        if len(keys) != len(self._by_id):
            raise ValueError("duplicate model_condition_key in ontology")

    def get(self, disability_id: str) -> DisabilityClass:
        """Return disability class by id."""
        if disability_id not in self._by_id:
            raise KeyError(disability_id)
        return self._by_id[disability_id]

    def therapy_focus_for(self, disability_id: str) -> list[str]:
        """Return recommended therapy task types for a disability."""
        return list(self.get(disability_id).therapy_focus)

    def to_dict(self) -> dict[str, Any]:
        """Serialize ontology summary for API traces."""
        return {
            "version": self.version,
            "disability_count": len(self._by_id),
            "ids": sorted(self._by_id.keys()),
        }
