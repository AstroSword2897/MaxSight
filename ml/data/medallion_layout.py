"""Bronze / silver / gold paths and training index for COCO + optional video manifests.

Layers
------
Bronze  Raw data exactly as obtained (no modification).
Silver  Cleaned, validated, split data ready for DataLoaders.
Gold    training_index.json — single resolved pointer for training scripts.

Supported datasets
------------------
coco            COCO 2017 detection / panoptic (default supervised)
kinetics700     Kinetics-700 action clips
youtube8m       YouTube-8M weak-label segments
howto100m       HowTo100M instructional video
webvid10m       WebVid-10M video-text pairs
bdd100k         BDD100K driving (detection, tracking, seg)
epic_kitchens   Epic-Kitchens-100 egocentric
mose            MOSE video object segmentation
youtube_vos     YouTube-VOS video object segmentation
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


MEDALLION_INDEX_VERSION = "1.0"

# Canonical dataset keys used everywhere.
DATASET_KEYS: List[str] = [
    "coco",
    "kinetics700",
    "youtube8m",
    "howto100m",
    "webvid10m",
    "bdd100k",
    "epic_kitchens",
    "mose",
    "youtube_vos",
]


def default_medallion_root(repo_root: Path) -> Path:
    return Path(repo_root).resolve() / "datasets" / "medallion"


# Stable roots for vendor drops; scripts and ingest should prefer these over ad-hoc paths.
RAW_DATASETS_DIRNAME = "raw"


def default_raw_dataset_dir(repo_root: Path, dataset_key: str) -> Path:
    """Return the canonical directory for raw dataset files.

    COCO stays under ``datasets/coco_raw`` to match existing download and gather scripts.
    All other keys use ``datasets/raw/<dataset_key>/``.
    """

    rr = Path(repo_root).resolve()
    if dataset_key not in DATASET_KEYS:
        raise ValueError(f"Unknown dataset key {dataset_key!r}. Valid: {DATASET_KEYS}")
    if dataset_key == "coco":
        return rr / "datasets" / "coco_raw"
    return rr / "datasets" / RAW_DATASETS_DIRNAME / dataset_key


def all_default_raw_dataset_dirs(repo_root: Path) -> Dict[str, Path]:
    """Map every ``DATASET_KEYS`` entry to its canonical raw path."""

    return {k: default_raw_dataset_dir(repo_root, k) for k in DATASET_KEYS}


# ── Bronze paths ───────────────────────────────────────────────────────────────

def bronze_coco_dir(root: Path) -> Path:
    return Path(root) / "bronze" / "coco"


def bronze_video_dir(root: Path) -> Path:
    """Generic bronze video root (used for ad-hoc video drops)."""
    return Path(root) / "bronze" / "video"


def bronze_dataset_dir(root: Path, dataset_key: str) -> Path:
    """Per-dataset bronze directory."""
    if dataset_key not in DATASET_KEYS:
        raise ValueError(f"Unknown dataset key {dataset_key!r}. Valid: {DATASET_KEYS}")
    return Path(root) / "bronze" / dataset_key


# ── Silver paths ───────────────────────────────────────────────────────────────

def silver_coco_splits_dir(root: Path) -> Path:
    return Path(root) / "silver" / "coco" / "splits"


def silver_video_dir(root: Path) -> Path:
    return Path(root) / "silver" / "video"


def silver_dataset_dir(root: Path, dataset_key: str) -> Path:
    """Per-dataset silver directory (cleaned / split output)."""
    if dataset_key not in DATASET_KEYS:
        raise ValueError(f"Unknown dataset key {dataset_key!r}. Valid: {DATASET_KEYS}")
    return Path(root) / "silver" / dataset_key


def silver_manifests_dir(root: Path, dataset_key: str) -> Path:
    """Where v1 clip manifests land after cleaning."""
    return silver_dataset_dir(root, dataset_key) / "manifests"


def silver_annotations_dir(root: Path, dataset_key: str) -> Path:
    """Where COCO-style annotation JSONs land after cleaning."""
    return silver_dataset_dir(root, dataset_key) / "annotations"


# ── Gold paths ─────────────────────────────────────────────────────────────────

def gold_dir(root: Path) -> Path:
    return Path(root) / "gold"


def default_gold_index_path(root: Path) -> Path:
    return gold_dir(root) / "training_index.json"


def path_relative_to_repo(path: Path, repo_root: Path) -> str:
    """Store repo-portable strings; fall back to absolute if outside repo."""

    pr = path.resolve()
    rr = repo_root.resolve()
    try:
        return str(pr.relative_to(rr))
    except ValueError:
        return str(pr)


def resolve_repo_path(repo_root: Path, stored: str) -> Path:
    p = Path(stored)
    if p.is_absolute():
        return p
    return (repo_root / p).resolve()


def build_coco_training_index(
    repo_root: Path,
    *,
    bronze_coco_data_dir: Path,
    train_annotation: Path,
    val_annotation: Path,
    test_annotation: Optional[Path] = None,
    image_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Assemble the `coco` section of a gold training index."""

    img = image_dir if image_dir is not None else bronze_coco_data_dir
    return {
        "schema_version": MEDALLION_INDEX_VERSION,
        "coco": {
            "data_dir": path_relative_to_repo(bronze_coco_data_dir, repo_root),
            "train_annotation": path_relative_to_repo(train_annotation, repo_root),
            "val_annotation": path_relative_to_repo(val_annotation, repo_root),
            "test_annotation": path_relative_to_repo(test_annotation, repo_root)
            if test_annotation is not None
            else None,
            "image_dir": path_relative_to_repo(img, repo_root),
        },
        "video": {
            "train_manifest": None,
            "val_manifest": None,
            "manifest_root": None,
        },
        "notes": "COCO silver splits reference bronze pixels; paths are relative to repo root unless absolute.",
    }


def merge_video_into_index(base: Dict[str, Any], video: Dict[str, Optional[str]]) -> Dict[str, Any]:
    """Attach non-null video paths into `video` section."""

    out = dict(base)
    v = dict(out.get("video") or {})
    for k, val in video.items():
        if val is not None:
            v[k] = val
    out["video"] = v
    return out


def write_training_index(path: Path, data: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_training_index(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def resolve_coco_for_train(
    index: Dict[str, Any], repo_root: Path
) -> Tuple[Path, Path, Path, Path]:
    """Return (data_dir, train_ann, val_ann, image_dir) as absolute paths."""

    coco = index.get("coco") or {}
    required = ("data_dir", "train_annotation", "val_annotation", "image_dir")
    for k in required:
        if not coco.get(k):
            raise KeyError(f"gold index missing coco.{k}")
    rr = Path(repo_root).resolve()
    return (
        resolve_repo_path(rr, coco["data_dir"]),
        resolve_repo_path(rr, coco["train_annotation"]),
        resolve_repo_path(rr, coco["val_annotation"]),
        resolve_repo_path(rr, coco["image_dir"]),
    )


def resolve_video_manifests(
    index: Dict[str, Any], repo_root: Path
) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """Return (train_manifest, val_manifest, manifest_root) or None entries."""

    vid = index.get("video") or {}
    rr = Path(repo_root).resolve()

    def _p(key: str) -> Optional[Path]:
        s = vid.get(key)
        if not s:
            return None
        return resolve_repo_path(rr, str(s))

    return _p("train_manifest"), _p("val_manifest"), _p("manifest_root")


def ensure_medallion_dirs(root: Path, datasets: Optional[List[str]] = None) -> None:
    """Create the full bronze/silver/gold directory tree for listed datasets."""

    keys = datasets if datasets is not None else DATASET_KEYS
    for k in keys:
        bronze_dataset_dir(root, k).mkdir(parents=True, exist_ok=True)
        silver_manifests_dir(root, k).mkdir(parents=True, exist_ok=True)
        silver_annotations_dir(root, k).mkdir(parents=True, exist_ok=True)
    gold_dir(root).mkdir(parents=True, exist_ok=True)


def dataset_bronze_manifest(root: Path, dataset_key: str) -> Path:
    """Path to the per-dataset bronze ingest record (JSON)."""
    return bronze_dataset_dir(root, dataset_key) / "ingest_record.json"


def write_ingest_record(
    root: Path,
    dataset_key: str,
    record: Dict[str, Any],
) -> Path:
    """Persist an ingest record to bronze so cleaning knows what arrived."""
    p = dataset_bronze_manifest(root, dataset_key)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return p


def load_ingest_record(root: Path, dataset_key: str) -> Dict[str, Any]:
    p = dataset_bronze_manifest(root, dataset_key)
    if not p.exists():
        raise FileNotFoundError(f"No ingest record for {dataset_key} at {p}")
    with open(p, encoding="utf-8") as f:
        return json.load(f)
