"""Load canonical label-space definitions used by the dataset registry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

LABEL_SPACE_REGISTRY_SCHEMA_VERSION = "1.0.0"


class LabelSpaceRegistryError(ValueError):
    """Raised when label_spaces.yaml is missing or invalid."""


@dataclass(frozen=True)
class LabelSpaceDefinition:
    """One named class vocabulary (immutable after load)."""

    id: str
    num_classes: int
    parent: str | None


@dataclass(frozen=True)
class LabelSpaceRegistry:
    """Loaded label-space table; resolve() is the only lookup path."""

    schema_version: str
    spaces: dict[str, LabelSpaceDefinition]
    source_path: str | None = None

    def resolve(self, label_space_id: str) -> LabelSpaceDefinition:
        if label_space_id not in self.spaces:
            raise LabelSpaceRegistryError(
                f"unknown label_space {label_space_id!r}; "
                f"define it in {self.source_path or '<label_spaces.yaml>'}"
            )
        return self.spaces[label_space_id]


def default_label_space_registry_path(repo_root: Path) -> Path:
    return (
        Path(repo_root).resolve() / "ml" / "training" / "configs" / "registry" / "label_spaces.yaml"
    )


def load_label_space_registry(path: Path | None = None) -> LabelSpaceRegistry:
    """Load and validate label_spaces.yaml."""
    if path is None:
        repo_root = Path(__file__).resolve().parents[2]
        path = default_label_space_registry_path(repo_root)
    path = Path(path)
    if not path.is_file():
        raise LabelSpaceRegistryError(f"label space registry not found at {path}")

    try:
        import yaml
    except ImportError as exc:
        raise LabelSpaceRegistryError(
            "PyYAML is required to load label space registry (pip install pyyaml)"
        ) from exc

    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise LabelSpaceRegistryError(f"root must be a mapping, got {type(raw).__name__}")
    schema = raw.get("schema_version")
    if schema != LABEL_SPACE_REGISTRY_SCHEMA_VERSION:
        raise LabelSpaceRegistryError(
            f"label_spaces schema_version {schema!r} != expected "
            f"{LABEL_SPACE_REGISTRY_SCHEMA_VERSION!r}"
        )
    items = raw.get("label_spaces")
    if not isinstance(items, list) or not items:
        raise LabelSpaceRegistryError("label_spaces must be a non-empty list")

    spaces: dict[str, LabelSpaceDefinition] = {}
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise LabelSpaceRegistryError(
                f"label_spaces[{idx}] must be a mapping, got {type(item).__name__}"
            )
        ls_id = item.get("id")
        if not isinstance(ls_id, str) or not ls_id.strip():
            raise LabelSpaceRegistryError(f"label_spaces[{idx}].id must be a non-empty string")
        allowed = {"id", "num_classes", "parent"}
        unknown = set(item.keys()) - allowed
        if unknown:
            raise LabelSpaceRegistryError(f"label_spaces[{idx}] unknown keys {sorted(unknown)}")
        nc = item.get("num_classes")
        if not isinstance(nc, int) or nc < 1:
            raise LabelSpaceRegistryError(f"label_spaces[{idx}].num_classes must be int >= 1")
        parent = item.get("parent")
        if parent is not None and (not isinstance(parent, str) or not parent.strip()):
            raise LabelSpaceRegistryError(
                f"label_spaces[{idx}].parent must be null or a non-empty string"
            )
        if ls_id in spaces:
            raise LabelSpaceRegistryError(f"duplicate label_space id {ls_id!r}")
        spaces[ls_id] = LabelSpaceDefinition(
            id=ls_id,
            num_classes=nc,
            parent=parent if isinstance(parent, str) else None,
        )

    return LabelSpaceRegistry(
        schema_version=schema,
        spaces=spaces,
        source_path=str(path),
    )


__all__ = [
    "LABEL_SPACE_REGISTRY_SCHEMA_VERSION",
    "LabelSpaceDefinition",
    "LabelSpaceRegistry",
    "LabelSpaceRegistryError",
    "default_label_space_registry_path",
    "load_label_space_registry",
]
