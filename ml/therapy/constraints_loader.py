"""Load therapy constraint YAML for runtime enforcement."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "config" / "therapy_constraints.yaml"


@dataclass(frozen=True)
class TherapyConstraints:
    """Executable therapy policy constraints."""

    version: str
    max_prompts_per_minute: float
    min_gap_between_prompts_s: float
    suppress_threshold: float
    disallowed_phrases: list[str]
    scoring_weights: dict[str, float]
    disability_routing: dict[str, list[str]]

    @classmethod
    def load(cls, path: Path | None = None) -> TherapyConstraints:
        """Parse constraints YAML."""
        p = path or DEFAULT_PATH
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        rate = raw.get("rate_limits", {})
        unc = raw.get("uncertainty", {})
        contra = raw.get("contraindications", {})
        return cls(
            version=str(raw.get("version", "0")),
            max_prompts_per_minute=float(rate.get("max_prompts_per_minute", 2)),
            min_gap_between_prompts_s=float(rate.get("min_gap_between_prompts_s", 10)),
            suppress_threshold=float(unc.get("suppress_threshold", 0.7)),
            disallowed_phrases=list(contra.get("disallowed_phrases", [])),
            scoring_weights=dict(raw.get("scoring_weights", {})),
            disability_routing=dict(raw.get("disability_routing", {})),
        )

    def is_disallowed_content(self, text: str) -> bool:
        """Return True when content matches a contraindicated phrase."""
        lower = text.lower()
        return any(phrase in lower for phrase in self.disallowed_phrases)
