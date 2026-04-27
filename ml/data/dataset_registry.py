"""Dataset registry: the only place a dataset becomes 'recognized' by the system.

Every dataset_id referenced by ResolvedTrainingConfig (and by extension every
tier YAML, CLI override, or SageMaker hyperparameter) must resolve to an entry
loaded from ``ml/training/configs/registry/datasets.yaml``. The registry is
strict: unknown ids, version mismatches, unknown fields, and (when present and
required) on-disk content_hash mismatches all raise DatasetRegistryError at
load time so silent dataset drift cannot reach training.

The registry never participates in deep merges. Composition layers reference
entries by id; a resolved DatasetEntry is treated as an atomic unit.

Why a separate module instead of folding into run_config.py:
  - run_config.py owns runtime config; the registry owns dataset identity.
  - Other tools (medallion ingest, gold-index builders, inference benchmarks)
    must consume the same recognition contract without pulling in training
    config types.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

logger = logging.getLogger(__name__)

REGISTRY_SCHEMA_VERSION = "1.2.0"

_ALLOWED_KIND = {"image", "video", "multimodal"}
_ALLOWED_STATUS = {"active", "registered", "deprecated", "raw"}
_ALLOWED_SOURCE = {"bronze", "silver", "gold", "external"}
_ALLOWED_DATASET_TYPES = {"real_world", "synthetic"}
# Canonical vocabularies are defined in ml/training/configs/registry/label_spaces.yaml.
# null means no MaxSight label contract yet (raw sources only).
_ALLOWED_CANONICAL_LABEL_SPACES = frozenset({"coco_80", "accessibility_622"})
# How annotations are encoded on disk; adapters normalize into the training sample dict.
_ALLOWED_ANNOTATION_FORMAT = {
    "maxsight_list",
    "coco_dict",
    "video_manifest",
    "multimodal_manifest",
}

# Mirror ml/data/medallion_layout.DATASET_KEYS without importing it, to keep
# this module dependency-free (the registry must load even if ml.data.* fails).
_ALLOWED_MEDALLION_KEYS = {
    "coco", "kinetics700", "youtube8m", "howto100m", "webvid10m",
    "bdd100k", "epic_kitchens", "mose", "youtube_vos",
}

# Mirror _ALLOWED_TIERS in run_config.py. Duplicated intentionally; both lists
# are short and the registry must not import the training config module.
_ALLOWED_TIERS = {
    "T0_BASELINE_CNN", "T1_LIGHTWEIGHT", "T2_DETECTOR", "T2_HYBRID_VIT",
    "T3_MULTI_TASK", "T4_ADVANCED", "T5_TEMPORAL",
}


# ── Errors ────────────────────────────────────────────────────────────────────

class DatasetRegistryError(ValueError):
    """Raised when the registry file or a resolution request violates contract."""


# ── Schema ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SplitSpec:
    """Annotations path + optional per-split image directory override.

    When image_dir is None the entry-level DatasetEntry.image_dir is used.
    This lets datasets with split-dependent image layouts (e.g. synthetic sets
    with train/val/test under separate image roots) be expressed without
    flattening the directory structure on disk.
    """
    annotations: str
    image_dir: Optional[str] = None


@dataclass(frozen=True)
class DatasetSplits:
    """Per-split specs; each value is a SplitSpec or None."""
    train: Optional[SplitSpec] = None
    val: Optional[SplitSpec] = None
    test: Optional[SplitSpec] = None


@dataclass(frozen=True)
class DatasetEntry:
    """One registered dataset; immutable once loaded."""
    id: str
    version: str
    kind: str
    status: str
    source: str
    dataset_type: str
    annotation_format: str
    # None = no canonical vocabulary contract (raw / uningested sources only).
    label_space: Optional[str]
    medallion_key: Optional[str]
    num_classes: int
    splits: Optional[DatasetSplits]
    # Fallback image root when a SplitSpec does not declare its own image_dir.
    image_dir: Optional[str]
    audio_dir: Optional[str]
    manifest: Optional[str]
    condition_modes_supported: Optional[List[str]]
    tier_compatibility: List[str]
    content_hash: Optional[str]
    notes: str = ""

    @property
    def key(self) -> str:
        """Composite (id, version) lookup key."""
        return f"{self.id}@{self.version}"

    def resolved_image_dir(self, split: str = "train") -> Optional[str]:
        """Return the image directory to use for a given split.

        Prefers the per-split image_dir from SplitSpec; falls back to the
        entry-level image_dir. Returns None if neither is set.
        """
        if self.splits is not None:
            spec: Optional[SplitSpec] = getattr(self.splits, split, None)
            if spec is not None and spec.image_dir is not None:
                return spec.image_dir
        return self.image_dir

    def annotation_path(self, split: str) -> Optional[str]:
        """Return the annotation file path for the given split, or None."""
        if self.splits is None:
            return None
        spec: Optional[SplitSpec] = getattr(self.splits, split, None)
        return spec.annotations if spec is not None else None


@dataclass(frozen=True)
class DatasetRegistry:
    """Loaded registry; resolve() is the only sanctioned access path."""
    schema_version: str
    entries: Dict[str, DatasetEntry] = field(default_factory=dict)
    source_path: Optional[str] = None

    def resolve(
        self,
        dataset_id: str,
        version: Optional[str] = None,
        *,
        tier: Optional[str] = None,
        require_active: bool = False,
    ) -> DatasetEntry:
        """Look up an entry by id (and optional version).

        Failure modes (all raise DatasetRegistryError):
          - dataset_id has no entries           -> "unknown dataset_id"
          - version given but no exact match    -> "registered but version X not found"
          - require_active and status != active -> "registered but inactive"
          - tier given and tier not in entry.tier_compatibility -> "tier not certified"
        """
        candidates = [e for e in self.entries.values() if e.id == dataset_id]
        if not candidates:
            raise DatasetRegistryError(
                f"unknown dataset_id {dataset_id!r}; "
                f"add an entry to {self.source_path or '<registry>'} before referencing it"
            )
        if version is None:
            actives = [e for e in candidates if e.status == "active"]
            entry = actives[0] if actives else candidates[0]
        else:
            matches = [e for e in candidates if e.version == version]
            if not matches:
                seen = sorted({e.version for e in candidates})
                raise DatasetRegistryError(
                    f"dataset {dataset_id!r} registered but version {version!r} not found; "
                    f"known versions: {seen}"
                )
            entry = matches[0]
        if require_active and entry.status != "active":
            raise DatasetRegistryError(
                f"dataset {entry.key!r} is {entry.status!r}, not 'active'; "
                f"activate it in the registry after ingest, or pick an active dataset"
            )
        if tier is not None and tier not in entry.tier_compatibility:
            raise DatasetRegistryError(
                f"dataset {entry.key!r} is not certified for tier {tier!r}; "
                f"certified tiers: {entry.tier_compatibility}"
            )
        return entry

    def all_ids(self) -> List[str]:
        return sorted({e.id for e in self.entries.values()})


# ── Loader ────────────────────────────────────────────────────────────────────

def default_registry_path(repo_root: Path) -> Path:
    """Canonical location of the registry file."""
    return Path(repo_root).resolve() / "ml" / "training" / "configs" / "registry" / "datasets.yaml"


def load_registry(path: Optional[Path] = None) -> DatasetRegistry:
    """Load and validate the registry; the only sanctioned constructor."""
    if path is None:
        repo_root = Path(__file__).resolve().parents[2]
        path = default_registry_path(repo_root)
    path = Path(path)
    if not path.is_file():
        raise DatasetRegistryError(f"Dataset registry not found at {path}")

    try:
        import yaml
    except ImportError as exc:
        raise DatasetRegistryError(
            "PyYAML is required to load the dataset registry (pip install pyyaml)"
        ) from exc

    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise DatasetRegistryError(
            f"registry root must be a mapping, got {type(raw).__name__}"
        )
    schema_version = raw.get("schema_version")
    if schema_version != REGISTRY_SCHEMA_VERSION:
        raise DatasetRegistryError(
            f"registry schema_version {schema_version!r} != expected {REGISTRY_SCHEMA_VERSION!r}"
        )
    entries_raw = raw.get("datasets")
    if not isinstance(entries_raw, list) or not entries_raw:
        raise DatasetRegistryError("registry.datasets must be a non-empty list")

    from ml.data.label_space_registry import load_label_space_registry

    label_spaces_path = path.parent / "label_spaces.yaml"
    label_spaces = load_label_space_registry(label_spaces_path)

    entries: Dict[str, DatasetEntry] = {}
    for idx, item in enumerate(entries_raw):
        if not isinstance(item, dict):
            raise DatasetRegistryError(
                f"datasets[{idx}] must be a mapping, got {type(item).__name__}"
            )
        entry = _build_entry(item, where=f"datasets[{idx}]", label_spaces=label_spaces)
        if entry.key in entries:
            raise DatasetRegistryError(f"duplicate registry entry {entry.key!r}")
        entries[entry.key] = entry

    return DatasetRegistry(
        schema_version=schema_version,
        entries=entries,
        source_path=str(path),
    )


# ── Internal: per-entry validation ────────────────────────────────────────────

_ENTRY_FIELDS = {f.name for f in fields(DatasetEntry)}
_REQUIRED_ENTRY_KEYS = {
    "id", "version", "kind", "status", "source", "dataset_type", "annotation_format",
    "label_space", "medallion_key", "splits", "image_dir", "audio_dir", "manifest",
    "num_classes", "condition_modes_supported", "tier_compatibility", "content_hash",
}
_OPTIONAL_ENTRY_KEYS = {"notes"}


def _build_entry(
    raw: Mapping[str, Any],
    *,
    where: str,
    label_spaces: Any,
) -> DatasetEntry:
    unknown = set(raw.keys()) - (_REQUIRED_ENTRY_KEYS | _OPTIONAL_ENTRY_KEYS)
    if unknown:
        raise DatasetRegistryError(f"{where}: unknown keys {sorted(unknown)}")
    missing = _REQUIRED_ENTRY_KEYS - set(raw.keys())
    if missing:
        raise DatasetRegistryError(f"{where}: missing required keys {sorted(missing)}")

    id_ = _str(raw["id"], where, "id")
    version = _str(raw["version"], where, "version")
    kind = _str(raw["kind"], where, "kind")
    if kind not in _ALLOWED_KIND:
        raise DatasetRegistryError(
            f"{where}.kind {kind!r} not in {sorted(_ALLOWED_KIND)}"
        )
    status = _str(raw["status"], where, "status")
    if status not in _ALLOWED_STATUS:
        raise DatasetRegistryError(
            f"{where}.status {status!r} not in {sorted(_ALLOWED_STATUS)}"
        )
    source = _str(raw["source"], where, "source")
    if source not in _ALLOWED_SOURCE:
        raise DatasetRegistryError(
            f"{where}.source {source!r} not in {sorted(_ALLOWED_SOURCE)}"
        )
    dataset_type = _str(raw["dataset_type"], where, "dataset_type")
    if dataset_type not in _ALLOWED_DATASET_TYPES:
        raise DatasetRegistryError(
            f"{where}.dataset_type {dataset_type!r} not in {sorted(_ALLOWED_DATASET_TYPES)}"
        )
    annotation_format = _str(raw["annotation_format"], where, "annotation_format")
    if annotation_format not in _ALLOWED_ANNOTATION_FORMAT:
        raise DatasetRegistryError(
            f"{where}.annotation_format {annotation_format!r} not in {sorted(_ALLOWED_ANNOTATION_FORMAT)}"
        )
    if kind == "image" and annotation_format not in {"maxsight_list", "coco_dict"}:
        raise DatasetRegistryError(
            f"{where}: kind=image requires annotation_format in {{maxsight_list, coco_dict}}"
        )
    if kind == "video" and annotation_format != "video_manifest":
        raise DatasetRegistryError(
            f"{where}: kind=video requires annotation_format=video_manifest"
        )
    if kind == "multimodal" and annotation_format != "multimodal_manifest":
        raise DatasetRegistryError(
            f"{where}: kind=multimodal requires annotation_format=multimodal_manifest"
        )

    ls_raw = raw.get("label_space")
    label_space: Optional[str]
    if ls_raw is None:
        label_space = None
    elif isinstance(ls_raw, str) and ls_raw.strip():
        label_space = ls_raw.strip()
        if label_space not in _ALLOWED_CANONICAL_LABEL_SPACES:
            raise DatasetRegistryError(
                f"{where}.label_space {label_space!r} not in {sorted(_ALLOWED_CANONICAL_LABEL_SPACES)}"
            )
    else:
        raise DatasetRegistryError(
            f"{where}.label_space must be null or a non-empty string"
        )

    if status == "raw" and label_space is not None:
        raise DatasetRegistryError(
            f"{where}: status='raw' requires label_space=null (undefined vocabulary until ingest)"
        )
    if status == "active" and kind in ("image", "multimodal") and label_space is None:
        raise DatasetRegistryError(
            f"{where}: active {kind} datasets must declare a canonical label_space"
        )

    medallion_key = raw["medallion_key"]
    if medallion_key is not None:
        medallion_key = _str(medallion_key, where, "medallion_key")
        if medallion_key not in _ALLOWED_MEDALLION_KEYS:
            raise DatasetRegistryError(
                f"{where}.medallion_key {medallion_key!r} not in {sorted(_ALLOWED_MEDALLION_KEYS)}"
            )

    num_classes = raw["num_classes"]
    if not isinstance(num_classes, int) or num_classes < 0:
        raise DatasetRegistryError(f"{where}.num_classes must be int >= 0")

    if label_space is not None:
        spec = label_spaces.resolve(label_space)
        if num_classes != spec.num_classes:
            raise DatasetRegistryError(
                f"{where}: label_space={label_space!r} requires num_classes={spec.num_classes} "
                f"per label_spaces.yaml, got num_classes={num_classes}"
            )

    splits = _build_splits(raw["splits"], where=where, status=status)
    image_dir = _opt_str(raw["image_dir"], where, "image_dir")
    audio_dir = _opt_str(raw["audio_dir"], where, "audio_dir")
    manifest = _opt_str(raw["manifest"], where, "manifest")

    condition_modes = raw["condition_modes_supported"]
    if condition_modes is not None:
        if not isinstance(condition_modes, list) or not all(isinstance(x, str) for x in condition_modes):
            raise DatasetRegistryError(
                f"{where}.condition_modes_supported must be null or list[str]"
            )

    tier_compat = raw["tier_compatibility"]
    if not isinstance(tier_compat, list) or not tier_compat:
        raise DatasetRegistryError(
            f"{where}.tier_compatibility must be a non-empty list"
        )
    bad_tiers = [t for t in tier_compat if t not in _ALLOWED_TIERS]
    if bad_tiers:
        raise DatasetRegistryError(
            f"{where}.tier_compatibility contains unknown tiers {bad_tiers}; "
            f"allowed: {sorted(_ALLOWED_TIERS)}"
        )

    content_hash = raw["content_hash"]
    if content_hash is not None:
        if not isinstance(content_hash, str) or len(content_hash) != 64:
            raise DatasetRegistryError(
                f"{where}.content_hash must be null or 64-char sha256 hex"
            )

    if status == "active" and (splits is None or splits.train is None or splits.val is None):
        raise DatasetRegistryError(
            f"{where}: active datasets must declare splits.train and splits.val"
        )

    return DatasetEntry(
        id=id_,
        version=version,
        kind=kind,
        status=status,
        source=source,
        dataset_type=dataset_type,
        annotation_format=annotation_format,
        label_space=label_space,
        medallion_key=medallion_key,
        num_classes=num_classes,
        splits=splits,
        image_dir=image_dir,
        audio_dir=audio_dir,
        manifest=manifest,
        condition_modes_supported=list(condition_modes) if condition_modes is not None else None,
        tier_compatibility=list(tier_compat),
        content_hash=content_hash,
        notes=str(raw.get("notes") or ""),
    )


def _build_splits(raw: Any, *, where: str, status: str) -> Optional[DatasetSplits]:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise DatasetRegistryError(f"{where}.splits must be null or a mapping")
    allowed_keys = {"train", "val", "test"}
    unknown = set(raw.keys()) - allowed_keys
    if unknown:
        raise DatasetRegistryError(
            f"{where}.splits has unknown keys {sorted(unknown)}; allowed: {sorted(allowed_keys)}"
        )
    return DatasetSplits(
        train=_build_split_spec(raw.get("train"), where=where, name="train"),
        val=_build_split_spec(raw.get("val"), where=where, name="val"),
        test=_build_split_spec(raw.get("test"), where=where, name="test"),
    )


def _build_split_spec(raw: Any, *, where: str, name: str) -> Optional[SplitSpec]:
    """Parse a split entry.

    Accepts two forms:
      - Mapping: {annotations: "path", image_dir: "optional/path"}
      - String: "path"  (treated as annotations; image_dir resolved from entry-level)
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        if not raw.strip():
            raise DatasetRegistryError(f"{where}.splits.{name} must be a non-empty string")
        return SplitSpec(annotations=raw)
    if isinstance(raw, dict):
        allowed = {"annotations", "image_dir"}
        unknown = set(raw.keys()) - allowed
        if unknown:
            raise DatasetRegistryError(
                f"{where}.splits.{name} has unknown keys {sorted(unknown)}; "
                f"allowed: {sorted(allowed)}"
            )
        if "annotations" not in raw:
            raise DatasetRegistryError(f"{where}.splits.{name} must have 'annotations' key")
        annotations = _str(raw["annotations"], where, f"splits.{name}.annotations")
        image_dir = _opt_str(raw.get("image_dir"), where, f"splits.{name}.image_dir")
        return SplitSpec(annotations=annotations, image_dir=image_dir)
    raise DatasetRegistryError(
        f"{where}.splits.{name} must be a string or mapping, got {type(raw).__name__}"
    )


def _str(value: Any, where: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetRegistryError(f"{where}.{field_name} must be a non-empty string")
    return value


def _opt_str(value: Any, where: str, field_name: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise DatasetRegistryError(f"{where}.{field_name} must be null or a non-empty string")
    return value


# ── Optional disk verification (called only when require_match=true) ──────────

def verify_content_hash(
    entry: DatasetEntry,
    *,
    repo_root: Path,
) -> None:
    """Recompute hash over the entry's split files; raise on mismatch.

    No-op when entry.content_hash is None or required files do not exist on
    disk (callers decide whether absence is itself a failure based on
    require_match semantics; the registry only enforces equality).
    """
    if entry.content_hash is None or entry.splits is None:
        return
    rr = Path(repo_root).resolve()
    paths: List[Path] = []
    for name in ("train", "val", "test"):
        spec: Optional[SplitSpec] = getattr(entry.splits, name, None)
        if spec is None:
            continue
        paths.append((rr / spec.annotations).resolve())
    existing = [p for p in paths if p.is_file()]
    if not existing:
        return
    h = hashlib.sha256()
    for p in sorted(existing, key=lambda x: str(x)):
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    digest = h.hexdigest()
    if digest != entry.content_hash:
        raise DatasetRegistryError(
            f"dataset {entry.key!r} content_hash mismatch: "
            f"expected {entry.content_hash[:12]}…, got {digest[:12]}… "
            f"(files: {[str(p) for p in existing]})"
        )
