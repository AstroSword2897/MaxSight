"""Class-space definition and per-source label harmonisation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "configs" / "data" / "label_map.yaml"


@dataclass(frozen=True)
class LabelMap:
    classes: tuple[str, ...]
    sources: dict[str, dict[str, str]]
    ignore: dict[str, frozenset[str]]
    version: str = ""  # schema_version of the label map file, recorded on every manifest record

    @classmethod
    def load(cls, path: Path | str = DEFAULT_PATH) -> LabelMap:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        classes = tuple(raw["classes"])
        sources = {src: dict(m) for src, m in raw["sources"].items()}
        for src, mapping in sources.items():
            unknown = set(mapping.values()) - set(classes)
            if unknown:
                raise ValueError(f"{src}: maps to classes not in the class space: {sorted(unknown)}")
        ignore = {src: frozenset(names) for src, names in raw.get("ignore", {}).items()}
        return cls(classes=classes, sources=sources, ignore=ignore, version=str(raw.get("schema_version", "")))

    def map(self, source: str, name: str) -> str | None:
        """Return the MaxSight class for a source class name, or None if unmapped."""
        return self.sources.get(source, {}).get(name)

    def is_ignored(self, source: str, name: str) -> bool:
        return name in self.ignore.get(source, frozenset())

    def index(self, class_name: str) -> int:
        return self.classes.index(class_name)
