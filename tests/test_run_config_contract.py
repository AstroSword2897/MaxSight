"""Contract tests for ml.training.run_config.ResolvedTrainingConfig.

These guard the single-source-of-truth invariants the AWS rollout depends
on: strict schema, explicit-override merge precedence, no hidden defaults
that affect data/training, and tier enum coverage matching live YAMLs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.models.maxsight_cnn import CapabilityTier  # noqa: E402
from ml.training.run_config import (  # noqa: E402
    _UNSET,
    ConfigValidationError,
    ResolvedTrainingConfig,
    cli_overrides_from_namespace,
)

CONFIGS = [
    "t0_baseline",
    "t1_attention",
    "t2_hybrid_vit",
    "t3_cross_task",
    "t4_cross_modal",
    "t5_temporal",
    "t5_sec",
]

BASELINE_OVERRIDES = {"run_id": "test", "experiment": "ci"}


def _config_path(stem: str) -> Path:
    return PROJECT_ROOT / "ml" / "training" / "configs" / f"{stem}.yaml"


def _load_yaml_dict(path: Path) -> dict:
    import yaml

    raw = yaml.safe_load(path.read_text())
    assert isinstance(raw, dict)
    return raw


def _dump_yaml(data: dict) -> str:
    import yaml

    text = yaml.safe_dump(data)
    assert isinstance(text, str)
    return text


# ── Tier enum coverage ────────────────────────────────────────────────────────


def test_capability_tier_covers_all_live_configs() -> None:
    declared_tiers = set()

    for stem in CONFIGS:
        raw = _load_yaml_dict(_config_path(stem))
        declared_tiers.add(raw["model"]["tier"])
    enum_names = {t.name for t in CapabilityTier}
    missing = declared_tiers - enum_names
    assert not missing, f"YAML tiers not present in CapabilityTier: {sorted(missing)}"


@pytest.mark.parametrize("stem", CONFIGS)
def test_each_tier_yaml_resolves(stem: str) -> None:
    cfg = ResolvedTrainingConfig.from_sources(
        _config_path(stem),
        cli_overrides=BASELINE_OVERRIDES,
    )
    assert cfg.model.tier in {t.name for t in CapabilityTier}
    assert cfg.provenance.config_hash
    assert cfg.provenance.yaml_source.endswith(f"{stem}.yaml")


# ── Merge precedence ──────────────────────────────────────────────────────────


def test_cli_overrides_take_precedence_over_yaml() -> None:
    cfg = ResolvedTrainingConfig.from_sources(
        _config_path("t5_temporal"),
        cli_overrides={**BASELINE_OVERRIDES, "data.batch_size": 99, "training.num_epochs": 77},
    )
    assert cfg.data.batch_size == 99
    assert cfg.training.num_epochs == 77
    assert cfg.provenance.cli_overrides["data.batch_size"] == 99


def test_sm_hyperparameters_override_yaml() -> None:
    cfg = ResolvedTrainingConfig.from_sources(
        _config_path("t5_temporal"),
        cli_overrides=BASELINE_OVERRIDES,
        sm_hyperparameters={"data.batch_size": 12},
    )
    assert cfg.data.batch_size == 12
    assert cfg.provenance.sm_overrides == {"data.batch_size": 12}


def test_unset_cli_flag_does_not_override_yaml() -> None:
    overrides = cli_overrides_from_namespace(
        type("NS", (), {"epochs": _UNSET, "batch_size": _UNSET})(),
        {"epochs": "training.num_epochs", "batch_size": "data.batch_size"},
    )
    assert overrides == {}
    cfg = ResolvedTrainingConfig.from_sources(
        _config_path("t5_temporal"),
        cli_overrides={**BASELINE_OVERRIDES, **overrides},
    )
    assert cfg.training.num_epochs == 150
    assert cfg.data.batch_size == 4


# ── Strict schema rejection ───────────────────────────────────────────────────


def test_unknown_top_level_key_rejected(tmp_path: Path) -> None:
    src = _config_path("t5_temporal").read_text() + "\nbogus_field: 1\n"
    bad = tmp_path / "bad.yaml"
    bad.write_text(src)
    with pytest.raises(ConfigValidationError, match="Unknown top-level keys"):
        ResolvedTrainingConfig.from_sources(bad, cli_overrides=BASELINE_OVERRIDES)


def test_unknown_section_key_rejected(tmp_path: Path) -> None:
    raw = _load_yaml_dict(_config_path("t5_temporal"))
    raw["training"]["bogus_inner"] = True
    bad = tmp_path / "bad.yaml"
    bad.write_text(_dump_yaml(raw))
    with pytest.raises(ConfigValidationError, match="Unknown keys in 'training'"):
        ResolvedTrainingConfig.from_sources(bad, cli_overrides=BASELINE_OVERRIDES)


def test_loss_weight_keys_must_match_active_heads(tmp_path: Path) -> None:
    raw = _load_yaml_dict(_config_path("t5_temporal"))
    raw["loss"]["loss_weights"]["motion"] = 0.5
    bad = tmp_path / "bad.yaml"
    bad.write_text(_dump_yaml(raw))
    with pytest.raises(ConfigValidationError, match="loss.loss_weights mismatch"):
        ResolvedTrainingConfig.from_sources(bad, cli_overrides=BASELINE_OVERRIDES)


def test_temporal_supervision_must_match_model(tmp_path: Path) -> None:
    raw = _load_yaml_dict(_config_path("t5_temporal"))
    raw["loss"]["temporal_supervision"] = False
    bad = tmp_path / "bad.yaml"
    bad.write_text(_dump_yaml(raw))
    with pytest.raises(ConfigValidationError, match="temporal_supervision must equal"):
        ResolvedTrainingConfig.from_sources(bad, cli_overrides=BASELINE_OVERRIDES)


def test_fp16_on_cpu_rejected(tmp_path: Path) -> None:
    raw = _load_yaml_dict(_config_path("t5_temporal"))
    raw["training"]["mixed_precision"] = True
    raw["device"] = "cpu"
    bad = tmp_path / "bad.yaml"
    bad.write_text(_dump_yaml(raw))
    with pytest.raises(ConfigValidationError, match="mixed_precision=True"):
        ResolvedTrainingConfig.from_sources(bad, cli_overrides=BASELINE_OVERRIDES)


def test_warmup_must_be_strictly_less_than_epochs(tmp_path: Path) -> None:
    raw = _load_yaml_dict(_config_path("t5_temporal"))
    raw["training"]["warmup_epochs"] = raw["training"]["num_epochs"]
    bad = tmp_path / "bad.yaml"
    bad.write_text(_dump_yaml(raw))
    with pytest.raises(ConfigValidationError, match="warmup_epochs"):
        ResolvedTrainingConfig.from_sources(bad, cli_overrides=BASELINE_OVERRIDES)


def test_dataset_section_required(tmp_path: Path) -> None:
    raw = _load_yaml_dict(_config_path("t5_temporal"))
    raw.pop("dataset")
    bad = tmp_path / "bad.yaml"
    bad.write_text(_dump_yaml(raw))
    with pytest.raises(ConfigValidationError, match="dataset"):
        ResolvedTrainingConfig.from_sources(bad, cli_overrides=BASELINE_OVERRIDES)


# ── Hidden-default regression guards ──────────────────────────────────────────


def test_lighting_flags_must_be_explicit(tmp_path: Path) -> None:
    raw = _load_yaml_dict(_config_path("t5_temporal"))
    # Both lighting flags are required in data section; neither has a default.
    raw["data"].pop("tag_lighting_metadata")
    bad = tmp_path / "bad.yaml"
    bad.write_text(_dump_yaml(raw))
    with pytest.raises(ConfigValidationError, match="tag_lighting_metadata"):
        ResolvedTrainingConfig.from_sources(bad, cli_overrides=BASELINE_OVERRIDES)


def test_loss_weights_must_be_present_and_nonempty(tmp_path: Path) -> None:
    raw = _load_yaml_dict(_config_path("t5_temporal"))
    raw["loss"]["loss_weights"] = {}
    bad = tmp_path / "bad.yaml"
    bad.write_text(_dump_yaml(raw))
    with pytest.raises(ConfigValidationError, match="loss_weights"):
        ResolvedTrainingConfig.from_sources(bad, cli_overrides=BASELINE_OVERRIDES)


# ── Provenance ────────────────────────────────────────────────────────────────


def test_config_hash_changes_with_meaningful_overrides() -> None:
    a = ResolvedTrainingConfig.from_sources(
        _config_path("t5_temporal"),
        cli_overrides=BASELINE_OVERRIDES,
    )
    b = ResolvedTrainingConfig.from_sources(
        _config_path("t5_temporal"),
        cli_overrides={**BASELINE_OVERRIDES, "training.num_epochs": 200},
    )
    assert a.provenance.config_hash != b.provenance.config_hash


def test_canonical_dict_round_trip_is_jsonable() -> None:
    import json

    cfg = ResolvedTrainingConfig.from_sources(
        _config_path("t5_temporal"),
        cli_overrides=BASELINE_OVERRIDES,
    )
    blob = json.dumps(cfg.to_canonical_dict(), sort_keys=True, default=str)
    parsed = json.loads(blob)
    assert parsed["provenance"]["config_hash"] == cfg.provenance.config_hash
    assert parsed["model"]["tier"] == cfg.model.tier
