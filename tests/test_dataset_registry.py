"""Dataset registry contract: every dataset_id used anywhere must be recognized.

The registry is the only sanctioned recognition surface; these tests make
silent dataset drift an immediate test failure rather than a runtime surprise.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.data.dataset_registry import (  # noqa: E402
    DatasetRegistry,
    DatasetRegistryError,
    REGISTRY_SCHEMA_VERSION,
    default_registry_path,
    load_registry,
    verify_content_hash,
)
from ml.training.run_config import (  # noqa: E402
    ConfigValidationError,
    ResolvedTrainingConfig,
)

REGISTRY_PATH = default_registry_path(PROJECT_ROOT)
CONFIG_DIR = PROJECT_ROOT / "ml" / "training" / "configs"
TIER_CONFIGS = sorted(
    p for p in CONFIG_DIR.glob("*.yaml")
    if p.stem not in {"t5_temporal_2phase", "t2_to_t5_transfer"}
)
BASELINE_OVERRIDES = {"run_id": "test-run", "experiment": "ci"}


# ── Registry file: shape + invariants ─────────────────────────────────────────

def test_registry_loads_from_canonical_path() -> None:
    reg = load_registry(REGISTRY_PATH)
    assert isinstance(reg, DatasetRegistry)
    assert reg.schema_version == REGISTRY_SCHEMA_VERSION
    assert reg.entries, "registry must contain at least one entry"


def test_registry_has_all_nine_medallion_keys_registered() -> None:
    reg = load_registry(REGISTRY_PATH)
    seen_medallion = {e.medallion_key for e in reg.entries.values() if e.medallion_key}
    expected = {
        "coco", "kinetics700", "youtube8m", "howto100m", "webvid10m",
        "bdd100k", "epic_kitchens", "mose", "youtube_vos",
    }
    assert expected <= seen_medallion, f"missing medallion keys in registry: {expected - seen_medallion}"


def test_active_datasets_declare_train_and_val_splits() -> None:
    reg = load_registry(REGISTRY_PATH)
    actives = [e for e in reg.entries.values() if e.status == "active"]
    assert actives, "registry must declare at least one active dataset"
    for entry in actives:
        assert entry.splits is not None, f"{entry.key}: active entry missing splits"
        assert entry.splits.train is not None, f"{entry.key}: active entry missing splits.train"
        assert entry.splits.val is not None, f"{entry.key}: active entry missing splits.val"
        # SplitSpec carries annotations path, not a raw string.
        assert entry.splits.train.annotations, f"{entry.key}: splits.train.annotations empty"
        assert entry.splits.val.annotations, f"{entry.key}: splits.val.annotations empty"


def test_active_maxsight_coco_cleaned_v1_is_present() -> None:
    """All current tier YAMLs reference this id+version; it must exist."""
    reg = load_registry(REGISTRY_PATH)
    entry = reg.resolve("maxsight-coco-cleaned", "v1", require_active=True)
    assert entry.kind == "image"
    assert entry.dataset_type == "real_world"
    assert entry.label_space == "accessibility_622"
    # Vocabulary is the full 622-class COCO_BASE + ACCESSIBILITY set defined
    # in ml/models/maxsight_cnn.COCO_CLASSES; the previous value of 80 was
    # wrong and allowed silent class collapse in training.
    assert entry.num_classes == 622
    assert entry.annotation_format == "maxsight_list"
    assert "T5_TEMPORAL" in entry.tier_compatibility


# ── Resolution: failure modes are explicit, never silent ──────────────────────

def test_resolve_unknown_id_raises() -> None:
    reg = load_registry(REGISTRY_PATH)
    with pytest.raises(DatasetRegistryError, match="unknown dataset_id"):
        reg.resolve("not-a-real-dataset", "v1")


def test_resolve_known_id_unknown_version_raises() -> None:
    reg = load_registry(REGISTRY_PATH)
    with pytest.raises(DatasetRegistryError, match="version 'v999' not found"):
        reg.resolve("maxsight-coco-cleaned", "v999")


def test_resolve_inactive_with_require_active_raises() -> None:
    reg = load_registry(REGISTRY_PATH)
    with pytest.raises(DatasetRegistryError, match="not 'active'"):
        reg.resolve("kinetics700", "v0", require_active=True)


def test_resolve_tier_incompatibility_raises() -> None:
    reg = load_registry(REGISTRY_PATH)
    with pytest.raises(DatasetRegistryError, match="not certified for tier"):
        reg.resolve("coco-bronze", "2017", tier="T5_TEMPORAL")


# ── End-to-end through ResolvedTrainingConfig ─────────────────────────────────

@pytest.mark.parametrize("config_path", TIER_CONFIGS, ids=lambda p: p.stem)
def test_every_tier_yaml_resolves_through_registry(config_path: Path) -> None:
    """Each shipped tier YAML must reference a registered + tier-certified dataset."""
    cfg = ResolvedTrainingConfig.from_sources(config_path, cli_overrides=BASELINE_OVERRIDES)
    assert cfg.dataset.dataset_id
    assert cfg.dataset.dataset_version


def test_unknown_dataset_id_in_yaml_fails_resolution(tmp_path: Path) -> None:
    base = yaml.safe_load((CONFIG_DIR / "t0_baseline.yaml").read_text())
    base["dataset"]["dataset_id"] = "ghost-dataset"
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(base))
    with pytest.raises(ConfigValidationError, match="unknown dataset_id"):
        ResolvedTrainingConfig.from_sources(bad, cli_overrides=BASELINE_OVERRIDES)


def test_tier_outside_compat_list_fails_resolution(tmp_path: Path) -> None:
    base = yaml.safe_load((CONFIG_DIR / "t5_temporal.yaml").read_text())
    base["dataset"]["dataset_id"] = "coco-bronze"
    base["dataset"]["dataset_version"] = "2017"
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(base))
    with pytest.raises(ConfigValidationError, match="not certified for tier 'T5_TEMPORAL'"):
        ResolvedTrainingConfig.from_sources(bad, cli_overrides=BASELINE_OVERRIDES)


def test_inactive_registered_dataset_fails_resolution(tmp_path: Path) -> None:
    """Reserved medallion keys (status=registered) must not be trainable yet."""
    base = yaml.safe_load((CONFIG_DIR / "t5_temporal.yaml").read_text())
    base["dataset"]["dataset_id"] = "kinetics700"
    base["dataset"]["dataset_version"] = "v0"
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(base))
    with pytest.raises(ConfigValidationError, match="not 'active'"):
        ResolvedTrainingConfig.from_sources(bad, cli_overrides=BASELINE_OVERRIDES)


# ── Registry file-level validation ────────────────────────────────────────────

def _write_registry(tmp_path: Path, entries: List[Dict[str, Any]]) -> Path:
    # load_registry expects label_spaces.yaml beside the registry file.
    ls_src = PROJECT_ROOT / "ml/training/configs/registry/label_spaces.yaml"
    (tmp_path / "label_spaces.yaml").write_text(ls_src.read_text())
    p = tmp_path / "registry.yaml"
    p.write_text(yaml.safe_dump({
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "datasets": entries,
    }))
    return p


def _entry(**overrides: Any) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "id": "test-ds",
        "version": "v1",
        "kind": "image",
        "status": "active",
        "source": "silver",
        "dataset_type": "real_world",
        "annotation_format": "coco_dict",
        "label_space": "coco_80",
        "medallion_key": "coco",
        "splits": {"train": "a.json", "val": "b.json"},
        "image_dir": "imgs",
        "audio_dir": None,
        "manifest": None,
        "num_classes": 80,
        "condition_modes_supported": None,
        "tier_compatibility": ["T0_BASELINE_CNN"],
        "content_hash": None,
    }
    base.update(overrides)
    return base


def test_registry_rejects_unknown_keys(tmp_path: Path) -> None:
    p = _write_registry(tmp_path, [_entry(extra_key="nope")])
    with pytest.raises(DatasetRegistryError, match="unknown keys"):
        load_registry(p)


def test_registry_rejects_bad_kind(tmp_path: Path) -> None:
    p = _write_registry(tmp_path, [_entry(kind="hologram")])
    with pytest.raises(DatasetRegistryError, match="kind 'hologram' not in"):
        load_registry(p)


def test_registry_rejects_bad_dataset_type(tmp_path: Path) -> None:
    p = _write_registry(tmp_path, [_entry(dataset_type="imaginary")])
    with pytest.raises(DatasetRegistryError, match="dataset_type 'imaginary' not in"):
        load_registry(p)


def test_registry_rejects_bad_label_space(tmp_path: Path) -> None:
    p = _write_registry(tmp_path, [_entry(label_space="coco_999")])
    with pytest.raises(DatasetRegistryError, match="label_space 'coco_999' not in"):
        load_registry(p)


def test_registry_rejects_label_space_num_classes_mismatch(tmp_path: Path) -> None:
    # accessibility_622 requires num_classes=622; using 80 is a contract violation.
    p = _write_registry(tmp_path, [_entry(label_space="accessibility_622", num_classes=80)])
    with pytest.raises(DatasetRegistryError, match="requires num_classes=622"):
        load_registry(p)


def test_raw_status_requires_null_label_space(tmp_path: Path) -> None:
    p = _write_registry(tmp_path, [_entry(
        status="raw",
        kind="video",
        annotation_format="video_manifest",
        label_space="coco_80",
        splits=None,
        manifest="datasets/medallion/silver/kinetics700/manifests/clips.json",
        num_classes=700,
        tier_compatibility=["T5_TEMPORAL"],
    )])
    with pytest.raises(DatasetRegistryError, match="status='raw' requires label_space=null"):
        load_registry(p)


def test_active_image_requires_non_null_label_space(tmp_path: Path) -> None:
    p = _write_registry(tmp_path, [_entry(label_space=None)])
    with pytest.raises(DatasetRegistryError, match="must declare a canonical label_space"):
        load_registry(p)


def test_registry_rejects_bad_tier(tmp_path: Path) -> None:
    p = _write_registry(tmp_path, [_entry(tier_compatibility=["T9_PLAID"])])
    with pytest.raises(DatasetRegistryError, match="unknown tiers"):
        load_registry(p)


def test_registry_rejects_active_without_splits(tmp_path: Path) -> None:
    p = _write_registry(tmp_path, [_entry(splits=None)])
    with pytest.raises(DatasetRegistryError, match="must declare splits.train"):
        load_registry(p)


def test_per_split_image_dir_resolves_correctly(tmp_path: Path) -> None:
    """SplitSpec image_dir overrides the entry-level image_dir via resolved_image_dir()."""
    p = _write_registry(tmp_path, [_entry(
        splits={
            "train": {"annotations": "train.json", "image_dir": "custom/train"},
            "val":   {"annotations": "val.json",   "image_dir": "custom/val"},
        },
        label_space="accessibility_622",
        num_classes=622,
    )])
    reg = load_registry(p)
    entry = reg.resolve("test-ds", "v1")
    assert entry.splits.train.image_dir == "custom/train"
    assert entry.splits.val.image_dir == "custom/val"
    assert entry.resolved_image_dir("train") == "custom/train"
    assert entry.resolved_image_dir("val") == "custom/val"
    # annotation_path helper
    assert entry.annotation_path("train") == "train.json"


def test_per_split_image_dir_falls_back_to_entry_level(tmp_path: Path) -> None:
    """When SplitSpec has no image_dir, resolved_image_dir returns entry-level image_dir."""
    p = _write_registry(tmp_path, [_entry(
        splits={"train": "train.json", "val": "val.json"},
        image_dir="shared/images",
    )])
    reg = load_registry(p)
    entry = reg.resolve("test-ds", "v1")
    assert entry.splits.train.image_dir is None
    assert entry.resolved_image_dir("train") == "shared/images"


def test_registry_rejects_duplicate_keys(tmp_path: Path) -> None:
    p = _write_registry(tmp_path, [_entry(), _entry()])
    with pytest.raises(DatasetRegistryError, match="duplicate registry entry"):
        load_registry(p)


def test_registry_rejects_bad_content_hash(tmp_path: Path) -> None:
    p = _write_registry(tmp_path, [_entry(content_hash="too-short")])
    with pytest.raises(DatasetRegistryError, match="64-char sha256 hex"):
        load_registry(p)


# ── On-disk hash verification (only triggered when require_match=True) ────────

def test_verify_content_hash_passes_when_match(tmp_path: Path) -> None:
    train = tmp_path / "datasets" / "train.json"
    val = tmp_path / "datasets" / "val.json"
    train.parent.mkdir(parents=True, exist_ok=True)
    train.write_bytes(b'{"split":"train"}')
    val.write_bytes(b'{"split":"val"}')
    h = hashlib.sha256()
    for p in sorted([train, val], key=lambda x: str(x)):
        with p.open("rb") as f:
            h.update(f.read())
    digest = h.hexdigest()
    p = _write_registry(tmp_path, [_entry(
        splits={"train": "datasets/train.json", "val": "datasets/val.json"},
        content_hash=digest,
    )])
    reg = load_registry(p)
    entry = reg.resolve("test-ds", "v1")
    verify_content_hash(entry, repo_root=tmp_path)


def test_verify_content_hash_raises_on_drift(tmp_path: Path) -> None:
    train = tmp_path / "datasets" / "train.json"
    train.parent.mkdir(parents=True, exist_ok=True)
    train.write_bytes(b'{"original":true}')
    p = _write_registry(tmp_path, [_entry(
        splits={"train": "datasets/train.json", "val": "datasets/val.json"},
        content_hash="0" * 64,
    )])
    reg = load_registry(p)
    entry = reg.resolve("test-ds", "v1")
    with pytest.raises(DatasetRegistryError, match="content_hash mismatch"):
        verify_content_hash(entry, repo_root=tmp_path)
