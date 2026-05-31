"""ResolvedTrainingConfig: the only source of truth for a training run.

Loaders for YAML, CLI, and SageMaker hyperparameters merge into a single
frozen object that downstream training code consumes. Nothing else in
training code is allowed to read raw YAML, argparse.Namespace, or
os.environ for training-affecting fields. Missing required fields raise
ConfigValidationError at load time so silent drift cannot occur.

Merge policy: yaml_base + explicit-only CLI/SM overrides (locked in with the
operator). CLI flags using the _UNSET sentinel are dropped before merge so
"empty CLI" means "YAML wins".

Provenance is loader-populated (config hash, git commit, runtime env) and
must be embedded in checkpoint metadata so dataset, config, and model stay
linked across local, container, and SageMaker runs.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

from ml.data.dataset_registry import (
    DatasetRegistry,
    DatasetRegistryError,
    default_registry_path,
    load_registry,
    verify_content_hash,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0.0"

# Sentinel for argparse defaults so callers can detect "user did not pass this".
# Routing _UNSET into cli_overrides drops the key, preserving the YAML value.
_UNSET = object()


# ── Errors ────────────────────────────────────────────────────────────────────


class ConfigValidationError(ValueError):
    """Raised when the resolved config violates the schema contract."""


# ── Allowed value sets ────────────────────────────────────────────────────────

_ALLOWED_TIERS = {
    "T0_BASELINE_CNN",
    "T1_LIGHTWEIGHT",
    "T2_DETECTOR",
    "T2_HYBRID_VIT",
    "T3_MULTI_TASK",
    "T4_ADVANCED",
    "T5_TEMPORAL",
}
_ALLOWED_DEVICES = {"cuda", "cpu", "auto"}
_ALLOWED_OPTIMIZERS = {"AdamW"}
_ALLOWED_SCHEDULERS = {"cosine", "onecycle", "cosine_restarts"}
_ALLOWED_ES_METRICS = {"val_loss", "val_map"}
_ALLOWED_DATA_PLANES = frozenset({"legacy", "gold"})
_ALLOWED_TRAINING_LABEL_SPACES = frozenset({"accessibility_622"})
_TIERS_REQUIRING_ACCESSIBILITY_LABEL_SPACE = {
    "T3_MULTI_TASK",
    "T4_ADVANCED",
    "T5_TEMPORAL",
}
_ALLOWED_CONDITION_MODES = {
    None,
    "glaucoma",
    "amd",
    "cataracts",
    "color_blindness",
    "diabetic_retinopathy",
    "retinitis_pigmentosa",
    "cvi",
    "amblyopia",
    "strabismus",
    "refractive_errors",
    "myopia",
    "hyperopia",
    "astigmatism",
    "presbyopia",
}


# ── Schema sections ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ModelSection:
    tier: str
    num_classes: int
    use_se_attention: bool
    use_cbam_attention: bool
    use_hybrid_backbone: bool
    use_dynamic_conv: bool
    use_cross_task_attention: bool
    use_cross_modal_attention: bool
    use_temporal_modeling: bool
    use_retrieval: bool


@dataclass(frozen=True)
class DataSection:
    train_annotation_file: str
    val_annotation_file: str
    image_dir: str
    batch_size: int
    num_workers: int
    pin_memory: bool
    max_objects: int
    tag_lighting_metadata: bool
    lighting_pixel_augmentation: bool
    use_weighted_sampling: bool
    test_annotation_file: str | None = None
    audio_dir: str | None = None
    condition_mode: str | None = None
    class_weights: dict[int, float] | None = None
    shuffle_train: bool = True
    drop_last: bool = False
    video_clip_manifest: str | None = None
    data_plane: str = "legacy"
    gold_train_shard_paths: tuple[str, ...] | None = None
    gold_val_shard_paths: tuple[str, ...] | None = None
    gold_test_shard_paths: tuple[str, ...] | None = None
    # Meta-driven gold mode: paths to artifact meta.json files.  When these are
    # set, num_classes and label_space are derived from the artifact — no registry
    # lookup is required at runtime.
    gold_train_meta: str | None = None
    gold_val_meta: str | None = None
    gold_test_meta: str | None = None


@dataclass(frozen=True)
class TrainingSection:
    num_epochs: int
    optimizer: str
    learning_rate: float
    weight_decay: float
    scheduler_type: str
    warmup_epochs: int
    min_lr: float
    gradient_clip_norm: float
    gradient_accumulation_steps: int
    mixed_precision: bool
    freeze_backbone: bool
    freeze_backbone_epochs: int
    early_stopping_patience: int
    early_stopping_metric: str
    learning_rate_backbone: float | None = None
    learning_rate_head: float | None = None
    ema_decay: float = 0.9999
    checkpoint_interval: int = 0
    label_space: str = "accessibility_622"
    use_compile: bool = False
    use_gradient_checkpointing: bool = False


@dataclass(frozen=True)
class DistributedSection:
    """Distributed training configuration."""

    backend: str = "none"
    world_size: int = 1
    local_rank: int = 0


@dataclass(frozen=True)
class LossSection:
    use_gradnorm: bool
    loss_weights: dict[str, float]
    temporal_supervision: bool
    gradnorm_alpha: float = 0.0
    gradnorm_update_interval: int = 0


@dataclass(frozen=True)
class ValidationSection:
    val_check_interval: float
    monitor: str
    mode: str
    save_top_k: int = 1


@dataclass(frozen=True)
class CheckpointSection:
    save_dir: str
    save_last: bool = True
    save_every_n_epochs: int = 0


@dataclass(frozen=True)
class LoggingSection:
    log_dir: str
    log_every_n_steps: int = 50
    tensorboard: bool = False


@dataclass(frozen=True)
class TargetMetricsSection:
    mAP_50: float | None = None
    mAP_75: float | None = None


@dataclass(frozen=True)
class DatasetSourceRef:
    """One weighted source in a multi-dataset training mix."""

    dataset_id: str
    dataset_version: str
    weight: float


@dataclass(frozen=True)
class DatasetSection:
    """Dataset version gate; ties model<->dataset<->config in provenance.

    Either ``sources`` (composition) or the pair (dataset_id, dataset_version) is
    set. When ``sources`` is non-empty, weights must sum to 1.0 and every source
    must share the same label_space and annotation_format.
    """

    require_match: bool
    dataset_id: str | None = None
    dataset_version: str | None = None
    sources: tuple[DatasetSourceRef, ...] | None = None
    content_hash: str | None = None
    manifest_uri: str | None = None


@dataclass(frozen=True)
class ProvenanceSection:
    config_hash: str
    yaml_source: str
    cli_overrides: dict[str, Any]
    sm_overrides: dict[str, Any]
    git_commit: str | None
    git_dirty: bool
    resolved_at: str
    schema_version: str
    runtime_env: dict[str, str]


@dataclass(frozen=True)
class ResolvedTrainingConfig:
    """Single source of truth for a training run; constructed by from_sources()."""

    schema_version: str
    run_id: str
    experiment: str
    seed: int
    device: str
    model: ModelSection
    data: DataSection
    training: TrainingSection
    loss: LossSection
    validation: ValidationSection
    checkpoint: CheckpointSection
    logging_: LoggingSection
    target_metrics: TargetMetricsSection
    dataset: DatasetSection
    provenance: ProvenanceSection
    distributed: DistributedSection = field(default_factory=lambda: DistributedSection())

    # ── Construction ─────────────────────────────────────────────────────────

    @classmethod
    def from_sources(
        cls,
        yaml_path: Path,
        cli_overrides: Mapping[str, Any] | None = None,
        sm_hyperparameters: Mapping[str, Any] | None = None,
        *,
        registry_path: Path | None = None,
        verify_dataset_on_disk: bool = False,
    ) -> ResolvedTrainingConfig:
        """Load YAML, apply explicit-only overrides, validate, freeze, return.

        Raises ConfigValidationError if any required field is missing, any
        unknown key is present, or any cross-field invariant is violated.

        The dataset section is resolved against the registry at
        ``registry_path`` (default ``ml/training/configs/registry/datasets.yaml``);
        an unknown dataset_id, version mismatch, or tier-incompatibility
        raises ConfigValidationError. When ``verify_dataset_on_disk=True``
        and the registry entry has a content_hash, the hash is recomputed
        over the on-disk splits and a mismatch aborts the run.
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.is_file():
            raise ConfigValidationError(f"Config YAML not found: {yaml_path}")
        try:
            import yaml
        except ImportError as exc:
            raise ConfigValidationError(
                "PyYAML is required to load training configs (pip install pyyaml)"
            ) from exc
        raw = yaml.safe_load(yaml_path.read_text()) or {}
        if not isinstance(raw, dict):
            raise ConfigValidationError(
                f"Top-level YAML must be a mapping, got {type(raw).__name__}"
            )

        cli_clean = _strip_unset(cli_overrides or {})
        sm_clean = _strip_unset(sm_hyperparameters or {})

        merged = _deep_copy_dict(raw)
        for dotted, value in cli_clean.items():
            _set_dotted(merged, dotted, value)
        for dotted, value in sm_clean.items():
            _set_dotted(merged, dotted, value)

        _reject_unknown_keys(merged)

        model = _build_model_section(merged.get("model", {}))
        data = _build_data_section(merged.get("data", {}))
        training = _build_training_section(merged.get("training", {}))
        loss_section = _build_loss_section(merged.get("loss", {}))
        validation = _build_validation_section(merged.get("validation", {}))
        checkpoint = _build_checkpoint_section(merged.get("checkpoint", {}))
        logging_section = _build_logging_section(merged.get("logging", {}))
        target_metrics = _build_target_metrics_section(merged.get("target_metrics", {}))
        dataset_section = _build_dataset_section(merged.get("dataset", {}))
        distributed = _build_distributed_section(merged.get("distributed", {}))

        seed = _required(merged, "seed", int)
        device = _required(merged, "device", str)
        run_id = _required(merged, "run_id", str)
        experiment = _required(merged, "experiment", str)
        schema_version = merged.get("schema_version", SCHEMA_VERSION)
        if schema_version != SCHEMA_VERSION:
            raise ConfigValidationError(
                f"Unsupported schema_version {schema_version!r}; expected {SCHEMA_VERSION!r}"
            )

        canonical = {
            "schema_version": schema_version,
            "run_id": run_id,
            "experiment": experiment,
            "seed": seed,
            "device": device,
            "model": _section_to_dict(model),
            "data": _section_to_dict(data),
            "training": _section_to_dict(training),
            "loss": _section_to_dict(loss_section),
            "validation": _section_to_dict(validation),
            "checkpoint": _section_to_dict(checkpoint),
            "logging": _section_to_dict(logging_section),
            "target_metrics": _section_to_dict(target_metrics),
            "dataset": _section_to_dict(dataset_section),
        }
        config_hash = hashlib.sha256(
            json.dumps(canonical, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

        git_commit, git_dirty = _git_state(yaml_path.parent)
        provenance = ProvenanceSection(
            config_hash=config_hash,
            yaml_source=str(yaml_path.resolve()),
            cli_overrides=dict(cli_clean),
            sm_overrides=dict(sm_clean),
            git_commit=git_commit,
            git_dirty=git_dirty,
            resolved_at=time.strftime("%Y-%m-%dT%H:%M:%S%z") or time.strftime("%Y-%m-%dT%H:%M:%S"),
            schema_version=schema_version,
            runtime_env=_runtime_env(),
        )

        resolved = cls(
            schema_version=schema_version,
            run_id=run_id,
            experiment=experiment,
            seed=seed,
            device=device,
            model=model,
            data=data,
            training=training,
            loss=loss_section,
            validation=validation,
            checkpoint=checkpoint,
            logging_=logging_section,
            target_metrics=target_metrics,
            dataset=dataset_section,
            provenance=provenance,
            distributed=distributed,
        )
        _validate_cross_fields(resolved)
        if _is_gold_meta_driven(resolved):
            # Artifact meta carries all invariants; no registry lookup needed.
            logger.info(
                "gold meta-driven mode: skipping dataset registry validation "
                "(train_meta=%s val_meta=%s)",
                resolved.data.gold_train_meta,
                resolved.data.gold_val_meta,
            )
        else:
            _validate_dataset_against_registry(
                resolved,
                registry_path=registry_path,
                verify_on_disk=verify_dataset_on_disk,
            )
        return resolved

    # ── Output / provenance ──────────────────────────────────────────────────

    def to_canonical_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict for logging, hashing, and checkpoint metadata."""
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "experiment": self.experiment,
            "seed": self.seed,
            "device": self.device,
            "model": _section_to_dict(self.model),
            "data": _section_to_dict(self.data),
            "training": _section_to_dict(self.training),
            "loss": _section_to_dict(self.loss),
            "validation": _section_to_dict(self.validation),
            "checkpoint": _section_to_dict(self.checkpoint),
            "logging": _section_to_dict(self.logging_),
            "target_metrics": _section_to_dict(self.target_metrics),
            "dataset": _section_to_dict(self.dataset),
            "distributed": _section_to_dict(self.distributed),
            "provenance": _section_to_dict(self.provenance),
        }

    def log_at_startup(self, log: logging.Logger | None = None) -> None:
        """Emit a single JSON line summarising the resolved config."""
        target = log or logger
        target.info(
            "resolved_training_config %s",
            json.dumps(self.to_canonical_dict(), sort_keys=True, default=str),
        )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _strip_unset(mapping: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in mapping.items() if v is not _UNSET}


def _deep_copy_dict(d: Any) -> Any:
    if isinstance(d, dict):
        return {k: _deep_copy_dict(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_deep_copy_dict(v) for v in d]
    return d


def _set_dotted(target: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cursor = target
    for key in parts[:-1]:
        nxt = cursor.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cursor[key] = nxt
        cursor = nxt
    cursor[parts[-1]] = value


def _section_to_dict(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _section_to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, dict):
        return {k: _section_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_section_to_dict(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


_KNOWN_TOP_LEVEL = {
    "schema_version",
    "run_id",
    "experiment",
    "seed",
    "device",
    "model",
    "data",
    "training",
    "loss",
    "validation",
    "checkpoint",
    "logging",
    "target_metrics",
    "dataset",
    "distributed",
}
_KNOWN_KEYS_BY_SECTION: dict[str, set] = {
    "model": {f.name for f in fields(ModelSection)},
    "data": {f.name for f in fields(DataSection)},
    "training": {f.name for f in fields(TrainingSection)},
    "loss": {f.name for f in fields(LossSection)},
    "validation": {f.name for f in fields(ValidationSection)},
    "checkpoint": {f.name for f in fields(CheckpointSection)},
    "logging": {f.name for f in fields(LoggingSection)},
    "target_metrics": {f.name for f in fields(TargetMetricsSection)},
    "dataset": {f.name for f in fields(DatasetSection)},
    "distributed": {f.name for f in fields(DistributedSection)},
}


def _reject_unknown_keys(merged: dict[str, Any]) -> None:
    extra_top = set(merged.keys()) - _KNOWN_TOP_LEVEL
    if extra_top:
        raise ConfigValidationError(f"Unknown top-level keys in config: {sorted(extra_top)}")
    for section, allowed in _KNOWN_KEYS_BY_SECTION.items():
        sec = merged.get(section, {})
        if not isinstance(sec, dict):
            raise ConfigValidationError(
                f"Section {section!r} must be a mapping, got {type(sec).__name__}"
            )
        unknown = set(sec.keys()) - allowed
        if unknown:
            raise ConfigValidationError(f"Unknown keys in '{section}': {sorted(unknown)}")


def _required(d: dict[str, Any], key: str, expected_type: type) -> Any:
    if key not in d:
        raise ConfigValidationError(f"Missing required field: {key!r}")
    value = d[key]
    if expected_type is float and isinstance(value, int):
        value = float(value)
    if not isinstance(value, expected_type):
        raise ConfigValidationError(
            f"Field {key!r} must be {expected_type.__name__}, got {type(value).__name__}"
        )
    return value


def _required_section(name: str, raw: dict[str, Any], schema_cls: type) -> dict[str, Any]:
    if not raw:
        raise ConfigValidationError(f"Missing required section: {name!r}")
    if not isinstance(raw, dict):
        raise ConfigValidationError(f"Section {name!r} must be a mapping, got {type(raw).__name__}")
    return raw


def _build_model_section(raw: dict[str, Any]) -> ModelSection:
    raw = _required_section("model", raw, ModelSection)
    required = [f.name for f in fields(ModelSection)]
    missing = [k for k in required if k not in raw]
    if missing:
        raise ConfigValidationError(f"model section missing required keys: {missing}")
    if raw["tier"] not in _ALLOWED_TIERS:
        raise ConfigValidationError(f"model.tier {raw['tier']!r} not in {sorted(_ALLOWED_TIERS)}")
    return ModelSection(**{k: raw[k] for k in required})


def _coerce_path_tuple(val: Any, field_name: str) -> tuple[str, ...]:
    if isinstance(val, str):
        return (val,)
    if isinstance(val, list):
        if not val:
            raise ConfigValidationError(f"{field_name} must be a non-empty list when set")
        return tuple(str(x) for x in val)
    raise ConfigValidationError(
        f"{field_name} must be a string, a list of strings, or null, got {type(val).__name__}"
    )


def _build_data_section(raw: dict[str, Any]) -> DataSection:
    raw = _required_section("data", raw, DataSection)
    cm = raw.get("condition_mode", None)
    if cm not in _ALLOWED_CONDITION_MODES:
        raise ConfigValidationError(
            f"data.condition_mode {cm!r} not in {sorted(str(x) for x in _ALLOWED_CONDITION_MODES)}"
        )
    for f in fields(DataSection):
        no_default = f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
        if no_default and f.name not in raw:
            raise ConfigValidationError(f"data section missing required key: {f.name!r}")
    kwargs: dict[str, Any] = {}
    for f in fields(DataSection):
        if f.name in raw:
            kwargs[f.name] = raw[f.name]
        elif f.default is not dataclasses.MISSING:
            kwargs[f.name] = f.default
        else:
            kwargs[f.name] = f.default_factory()
    if kwargs["data_plane"] not in _ALLOWED_DATA_PLANES:
        raise ConfigValidationError(
            f"data.data_plane must be one of {sorted(_ALLOWED_DATA_PLANES)}, got {kwargs['data_plane']!r}"
        )
    for key in ("gold_train_shard_paths", "gold_val_shard_paths", "gold_test_shard_paths"):
        v = kwargs.get(key)
        if v is not None:
            kwargs[key] = _coerce_path_tuple(v, f"data.{key}")
    for key in ("gold_train_meta", "gold_val_meta", "gold_test_meta"):
        v = kwargs.get(key)
        if v is not None and not isinstance(v, str):
            raise ConfigValidationError(f"data.{key} must be a string path or null")
    return DataSection(**kwargs)


def _build_training_section(raw: dict[str, Any]) -> TrainingSection:
    raw = dict(_required_section("training", raw, TrainingSection))
    raw.setdefault("label_space", "accessibility_622")
    for f in fields(TrainingSection):
        no_default = f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
        if no_default and f.name not in raw:
            raise ConfigValidationError(f"training section missing required key: {f.name!r}")
    if raw["optimizer"] not in _ALLOWED_OPTIMIZERS:
        raise ConfigValidationError(
            f"training.optimizer {raw['optimizer']!r} not in {sorted(_ALLOWED_OPTIMIZERS)}"
        )
    if raw["scheduler_type"] not in _ALLOWED_SCHEDULERS:
        raise ConfigValidationError(
            f"training.scheduler_type {raw['scheduler_type']!r} not in {sorted(_ALLOWED_SCHEDULERS)}"
        )
    if raw["early_stopping_metric"] not in _ALLOWED_ES_METRICS:
        raise ConfigValidationError(
            f"training.early_stopping_metric {raw['early_stopping_metric']!r} not in {sorted(_ALLOWED_ES_METRICS)}"
        )
    if raw["label_space"] not in _ALLOWED_TRAINING_LABEL_SPACES:
        raise ConfigValidationError(
            f"training.label_space {raw['label_space']!r} not in {sorted(_ALLOWED_TRAINING_LABEL_SPACES)}"
        )
    kwargs: dict[str, Any] = {}
    for f in fields(TrainingSection):
        if f.name in raw:
            kwargs[f.name] = raw[f.name]
        elif f.default is not dataclasses.MISSING:
            kwargs[f.name] = f.default
        else:
            kwargs[f.name] = f.default_factory()
    return TrainingSection(**kwargs)


_ALLOWED_DISTRIBUTED_BACKENDS = frozenset({"none", "ddp", "fsdp"})


def _build_distributed_section(raw: dict[str, Any]) -> DistributedSection:
    raw = dict(raw or {})
    kwargs: dict[str, Any] = {}
    for f in fields(DistributedSection):
        if f.name in raw:
            kwargs[f.name] = raw[f.name]
        elif f.default is not dataclasses.MISSING:
            kwargs[f.name] = f.default
        else:
            kwargs[f.name] = f.default_factory()
    if kwargs["backend"] not in _ALLOWED_DISTRIBUTED_BACKENDS:
        raise ConfigValidationError(
            f"distributed.backend {kwargs['backend']!r} not in {sorted(_ALLOWED_DISTRIBUTED_BACKENDS)}"
        )
    return DistributedSection(**kwargs)


def _build_loss_section(raw: dict[str, Any]) -> LossSection:
    raw = _required_section("loss", raw, LossSection)
    required_keys = ["use_gradnorm", "loss_weights", "temporal_supervision"]
    missing = [k for k in required_keys if k not in raw]
    if missing:
        raise ConfigValidationError(f"loss section missing required keys: {missing}")
    weights = raw["loss_weights"]
    if not isinstance(weights, dict) or not weights:
        raise ConfigValidationError("loss.loss_weights must be a non-empty mapping")
    for k, v in weights.items():
        if not isinstance(k, str):
            raise ConfigValidationError(
                f"loss.loss_weights key must be str, got {type(k).__name__}"
            )
        if not isinstance(v, (int, float)) or v < 0:
            raise ConfigValidationError(f"loss.loss_weights[{k}] must be non-negative number")
    accepted = {f.name for f in fields(LossSection)}
    return LossSection(**{k: raw[k] for k in raw.keys() if k in accepted})


def _build_validation_section(raw: dict[str, Any]) -> ValidationSection:
    raw = _required_section("validation", raw, ValidationSection)
    required_keys = ["val_check_interval", "monitor", "mode"]
    missing = [k for k in required_keys if k not in raw]
    if missing:
        raise ConfigValidationError(f"validation section missing required keys: {missing}")
    accepted = {f.name for f in fields(ValidationSection)}
    return ValidationSection(**{k: raw[k] for k in raw.keys() if k in accepted})


def _build_checkpoint_section(raw: dict[str, Any]) -> CheckpointSection:
    raw = _required_section("checkpoint", raw, CheckpointSection)
    if "save_dir" not in raw:
        raise ConfigValidationError("checkpoint section missing required key: 'save_dir'")
    accepted = {f.name for f in fields(CheckpointSection)}
    return CheckpointSection(**{k: raw[k] for k in raw.keys() if k in accepted})


def _build_logging_section(raw: dict[str, Any]) -> LoggingSection:
    raw = _required_section("logging", raw, LoggingSection)
    if "log_dir" not in raw:
        raise ConfigValidationError("logging section missing required key: 'log_dir'")
    accepted = {f.name for f in fields(LoggingSection)}
    return LoggingSection(**{k: raw[k] for k in raw.keys() if k in accepted})


def _build_target_metrics_section(raw: dict[str, Any]) -> TargetMetricsSection:
    if not isinstance(raw, dict):
        raise ConfigValidationError(f"target_metrics must be a mapping, got {type(raw).__name__}")
    accepted = {f.name for f in fields(TargetMetricsSection)}
    return TargetMetricsSection(**{k: raw[k] for k in raw if k in accepted})


def _build_dataset_section(raw: dict[str, Any]) -> DatasetSection:
    raw = _required_section("dataset", raw, DatasetSection)
    if "require_match" not in raw:
        raise ConfigValidationError("dataset section missing required key: 'require_match'")
    if not isinstance(raw["require_match"], bool):
        raise ConfigValidationError("dataset.require_match must be a boolean")

    src_raw = raw.get("sources")
    if src_raw is not None:
        if src_raw == []:
            src_raw = None
    if src_raw is not None:
        if not isinstance(src_raw, list):
            raise ConfigValidationError("dataset.sources must be a list or null")
        refs: list[DatasetSourceRef] = []
        for i, item in enumerate(src_raw):
            if not isinstance(item, dict):
                raise ConfigValidationError(
                    f"dataset.sources[{i}] must be a mapping, got {type(item).__name__}"
                )
            for k in ("dataset_id", "dataset_version", "weight"):
                if k not in item:
                    raise ConfigValidationError(f"dataset.sources[{i}] missing required key {k!r}")
            did = item["dataset_id"]
            dver = item["dataset_version"]
            w = item["weight"]
            if not isinstance(did, str) or not did.strip():
                raise ConfigValidationError(
                    f"dataset.sources[{i}].dataset_id must be a non-empty string"
                )
            if not isinstance(dver, str) or not dver.strip():
                raise ConfigValidationError(
                    f"dataset.sources[{i}].dataset_version must be a non-empty string"
                )
            if not isinstance(w, (int, float)) or w <= 0:
                raise ConfigValidationError(
                    f"dataset.sources[{i}].weight must be a positive number"
                )
            refs.append(DatasetSourceRef(dataset_id=did, dataset_version=dver, weight=float(w)))
        wsum = sum(r.weight for r in refs)
        if abs(wsum - 1.0) > 1e-4:
            raise ConfigValidationError(f"dataset.sources weights must sum to 1.0, got {wsum}")
        accepted = {f.name for f in fields(DatasetSection)}
        base = {k: raw[k] for k in raw.keys() if k in accepted and k != "sources"}
        base["sources"] = tuple(refs)
        return DatasetSection(**base)

    for k in ("dataset_id", "dataset_version"):
        if k not in raw or raw[k] is None or (isinstance(raw[k], str) and not raw[k].strip()):
            raise ConfigValidationError(f"dataset.{k} is required when dataset.sources is absent")
    accepted = {f.name for f in fields(DatasetSection)}
    return DatasetSection(**{k: raw[k] for k in raw.keys() if k in accepted})


def _is_gold_meta_driven(cfg: ResolvedTrainingConfig) -> bool:
    """True when gold mode is configured with artifact meta files (no registry needed)."""
    return cfg.data.data_plane == "gold" and bool(
        cfg.data.gold_train_meta or cfg.data.gold_val_meta
    )


def _validate_gold_artifact_meta(cfg: ResolvedTrainingConfig) -> None:
    """Validate gold artifact meta files against model/training invariants.

    Checks label_space and num_classes from each artifact against the config;
    does not touch the dataset registry.
    """
    from ml.data.gold.dataset import load_gold_meta

    meta_fields = [
        ("data.gold_train_meta", cfg.data.gold_train_meta),
        ("data.gold_val_meta", cfg.data.gold_val_meta),
        ("data.gold_test_meta", cfg.data.gold_test_meta),
    ]
    for label, meta_path in meta_fields:
        if not meta_path:
            continue
        p = Path(meta_path)
        if not p.is_file():
            raise ConfigValidationError(f"{label} is not an existing file: {p}")
        try:
            meta = load_gold_meta(p)
        except (ValueError, OSError) as exc:
            raise ConfigValidationError(f"{label} failed to load: {exc}") from exc

        meta_ls = meta.get("label_space", "")
        if meta_ls != cfg.training.label_space:
            raise ConfigValidationError(
                f"{label} label_space {meta_ls!r} does not match "
                f"training.label_space {cfg.training.label_space!r}"
            )
        meta_nc = int(meta.get("num_classes", 0))
        if meta_nc != cfg.model.num_classes:
            raise ConfigValidationError(
                f"{label} num_classes={meta_nc} does not match "
                f"model.num_classes={cfg.model.num_classes}"
            )

    # When meta is provided, shard paths are optional (derived from meta.shards
    # at runtime). When only shard paths are given (escape hatch), still check
    # that train + val shards are present.
    if not cfg.data.gold_train_meta:
        if not cfg.data.gold_train_shard_paths:
            raise ConfigValidationError(
                "data_plane=gold requires either data.gold_train_meta "
                "or data.gold_train_shard_paths"
            )
    if not cfg.data.gold_val_meta:
        if not cfg.data.gold_val_shard_paths:
            raise ConfigValidationError(
                "data_plane=gold requires either data.gold_val_meta or data.gold_val_shard_paths"
            )


def _validate_cross_fields(cfg: ResolvedTrainingConfig) -> None:
    if cfg.device not in _ALLOWED_DEVICES:
        raise ConfigValidationError(f"device {cfg.device!r} not in {sorted(_ALLOWED_DEVICES)}")
    if cfg.training.mixed_precision and cfg.device == "cpu":
        raise ConfigValidationError("training.mixed_precision=True requires device in {cuda, auto}")
    if cfg.training.warmup_epochs >= cfg.training.num_epochs:
        raise ConfigValidationError(
            "training.warmup_epochs must be strictly less than training.num_epochs"
        )
    if not cfg.training.freeze_backbone and cfg.training.freeze_backbone_epochs != 0:
        raise ConfigValidationError(
            "training.freeze_backbone_epochs must be 0 when freeze_backbone is False"
        )
    if cfg.training.freeze_backbone and cfg.training.freeze_backbone_epochs <= 0:
        raise ConfigValidationError(
            "training.freeze_backbone_epochs must be > 0 when freeze_backbone is True"
        )
    if cfg.loss.use_gradnorm:
        if cfg.loss.gradnorm_alpha <= 0 or cfg.loss.gradnorm_update_interval <= 0:
            raise ConfigValidationError(
                "loss.use_gradnorm=True requires gradnorm_alpha>0 and gradnorm_update_interval>0"
            )
    if cfg.loss.temporal_supervision != cfg.model.use_temporal_modeling:
        raise ConfigValidationError(
            "loss.temporal_supervision must equal model.use_temporal_modeling"
        )
    if cfg.loss.temporal_supervision:
        for required_head in ("temporal_consistency", "flicker"):
            if required_head not in cfg.loss.loss_weights:
                raise ConfigValidationError(
                    f"loss.temporal_supervision=True requires loss_weights[{required_head!r}]"
                )
    if cfg.dataset.sources and cfg.data.use_weighted_sampling:
        raise ConfigValidationError(
            "data.use_weighted_sampling cannot be used with dataset.sources composition; "
            "use a single dataset or disable weighted sampling."
        )
    if cfg.data.data_plane == "gold" and cfg.dataset.sources:
        raise ConfigValidationError(
            "data.data_plane=gold does not support dataset.sources composition yet."
        )
    if _is_gold_meta_driven(cfg):
        _validate_gold_artifact_meta(cfg)
    expected_heads = _expected_loss_heads(cfg)
    if expected_heads is not None:
        actual = set(cfg.loss.loss_weights.keys())
        missing = expected_heads - actual
        extra = actual - expected_heads
        if missing or extra:
            raise ConfigValidationError(
                f"loss.loss_weights mismatch with active heads. missing={sorted(missing)}, extra={sorted(extra)}"
            )


def _repo_root_from_registry(registry: DatasetRegistry) -> Path:
    """Resolve repo root from the canonical registry file path."""
    if registry.source_path:
        return Path(registry.source_path).resolve().parents[4]
    return Path.cwd()


def _validate_dataset_against_registry(
    cfg: ResolvedTrainingConfig,
    *,
    registry_path: Path | None,
    verify_on_disk: bool,
) -> None:
    """Resolve dataset(s) against the registry; raise on any mismatch.

    The registry is the single recognition surface for datasets: an id that
    is not registered is treated as a configuration error, not a soft
    warning, because silent dataset drift is the failure mode this guards.
    """
    try:
        registry = load_registry(registry_path)
    except DatasetRegistryError as exc:
        raise ConfigValidationError(f"dataset registry failed to load: {exc}") from exc

    repo_root = _repo_root_from_registry(registry)

    if cfg.dataset.sources:
        entries = []
        for src in cfg.dataset.sources:
            try:
                e = registry.resolve(
                    src.dataset_id,
                    src.dataset_version,
                    tier=cfg.model.tier,
                    require_active=True,
                )
            except DatasetRegistryError as exc:
                raise ConfigValidationError(str(exc)) from exc
            entries.append(e)
        ls0 = entries[0].label_space
        af0 = entries[0].annotation_format
        nc0 = entries[0].num_classes
        for e in entries[1:]:
            if e.label_space != ls0 or e.annotation_format != af0 or e.num_classes != nc0:
                raise ConfigValidationError(
                    "dataset.sources entries must share the same label_space, "
                    f"annotation_format, and num_classes; mismatch involving {e.key!r}"
                )
        if ls0 is None:
            raise ConfigValidationError(
                "dataset.sources cannot mix raw datasets without a label_space"
            )
        primary = entries[0]
    else:
        assert cfg.dataset.dataset_id is not None and cfg.dataset.dataset_version is not None
        try:
            primary = registry.resolve(
                cfg.dataset.dataset_id,
                cfg.dataset.dataset_version,
                tier=cfg.model.tier,
                require_active=True,
            )
        except DatasetRegistryError as exc:
            raise ConfigValidationError(str(exc)) from exc

    if primary.label_space is None:
        raise ConfigValidationError(
            f"dataset {primary.key!r} has no label_space; pick an active ingested dataset for training"
        )
    if cfg.model.tier in _TIERS_REQUIRING_ACCESSIBILITY_LABEL_SPACE:
        if primary.label_space != "accessibility_622":
            raise ConfigValidationError(
                f"model.tier {cfg.model.tier!r} requires label_space accessibility_622 "
                f"on the training dataset, got {primary.label_space!r}"
            )
    if cfg.model.num_classes != primary.num_classes:
        raise ConfigValidationError(
            f"model.num_classes={cfg.model.num_classes} does not match registry "
            f"num_classes={primary.num_classes} for {primary.key!r}"
        )

    if cfg.training.label_space != primary.label_space:
        raise ConfigValidationError(
            f"training.label_space {cfg.training.label_space!r} must match registry "
            f"label_space {primary.label_space!r} for {primary.key!r}"
        )

    if cfg.data.data_plane == "gold":
        if primary.annotation_format == "video_manifest":
            raise ConfigValidationError(
                "data.data_plane=gold does not support registry annotation_format=video_manifest yet."
            )
        for label, paths in (
            ("data.gold_train_shard_paths", cfg.data.gold_train_shard_paths),
            ("data.gold_val_shard_paths", cfg.data.gold_val_shard_paths),
        ):
            if not paths:
                raise ConfigValidationError(
                    f"{label} is required and must list at least one JSONL shard when data_plane=gold"
                )
            for rel in paths:
                p = (repo_root / rel).resolve()
                if not p.is_file():
                    raise ConfigValidationError(f"{label} entry is not an existing file: {p}")

    if not cfg.dataset.sources and cfg.data.data_plane != "gold":
        train_rel = primary.annotation_path("train")
        val_rel = primary.annotation_path("val")
        if train_rel:
            expected_train = (repo_root / train_rel).resolve()
            cfg_train = Path(cfg.data.train_annotation_file)
            if not cfg_train.is_absolute():
                cfg_train = (repo_root / cfg_train).resolve()
            else:
                cfg_train = cfg_train.resolve()
            if cfg_train != expected_train:
                raise ConfigValidationError(
                    f"data.train_annotation_file {cfg_train} does not match registry "
                    f"for {primary.key!r} ({expected_train})"
                )
        if val_rel:
            expected_val = (repo_root / val_rel).resolve()
            cfg_val = Path(cfg.data.val_annotation_file)
            if not cfg_val.is_absolute():
                cfg_val = (repo_root / cfg_val).resolve()
            else:
                cfg_val = cfg_val.resolve()
            if cfg_val != expected_val:
                raise ConfigValidationError(
                    f"data.val_annotation_file {cfg_val} does not match registry "
                    f"for {primary.key!r} ({expected_val})"
                )

    if cfg.dataset.require_match and verify_on_disk:
        if cfg.dataset.sources:
            for e in entries:
                try:
                    verify_content_hash(e, repo_root=repo_root)
                except DatasetRegistryError as exc:
                    raise ConfigValidationError(str(exc)) from exc
        else:
            try:
                verify_content_hash(primary, repo_root=repo_root)
            except DatasetRegistryError as exc:
                raise ConfigValidationError(str(exc)) from exc


def _expected_loss_heads(cfg: ResolvedTrainingConfig) -> set | None:
    """Return the head names required for the active model + temporal flag.

    Mirrors create_loss_fn() in scripts/ops/train_maxsight.py so YAML weights
    must declare exactly the heads that will exist at runtime. Returns None
    when the active head set cannot yet be derived (kept conservative until
    head selection moves into the schema).
    """
    base = {"objectness", "classification", "box", "distance", "urgency"}
    if cfg.loss.temporal_supervision:
        base |= {"temporal_consistency", "flicker"}
    return base


def _git_state(start: Path) -> tuple[str | None, bool]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(start),
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(start),
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return None, False
    if commit.returncode != 0:
        return None, False
    return commit.stdout.strip() or None, bool(dirty.stdout.strip())


def _runtime_env() -> dict[str, str]:
    env: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda_available"] = str(bool(torch.cuda.is_available()))
    except Exception:
        env["torch"] = "unavailable"
        env["cuda_available"] = "false"
    env["env_aws_region"] = os.environ.get("AWS_DEFAULT_REGION", "")
    env["env_sm_role"] = "set" if os.environ.get("SAGEMAKER_ROLE_ARN") else "unset"
    return env


# ── CLI helpers (used by ops scripts) ─────────────────────────────────────────


def cli_overrides_from_namespace(ns: Any, mapping: Mapping[str, str]) -> dict[str, Any]:
    """Build a dotted-key override dict from an argparse Namespace.

    mapping = {namespace_attr: dotted_config_path}. Values equal to _UNSET
    are dropped so YAML wins for any flag the operator did not pass.
    """
    out: dict[str, Any] = {}
    for attr, dotted in mapping.items():
        if not hasattr(ns, attr):
            continue
        val = getattr(ns, attr)
        if val is _UNSET:
            continue
        out[dotted] = val
    return out


__all__ = [
    "SCHEMA_VERSION",
    "_UNSET",
    "ConfigValidationError",
    "ModelSection",
    "DataSection",
    "TrainingSection",
    "LossSection",
    "ValidationSection",
    "CheckpointSection",
    "LoggingSection",
    "TargetMetricsSection",
    "DatasetSection",
    "ProvenanceSection",
    "ResolvedTrainingConfig",
    "cli_overrides_from_namespace",
    "DatasetRegistry",
    "DatasetRegistryError",
    "default_registry_path",
    "load_registry",
]
