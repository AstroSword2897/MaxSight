"""Model registry: version, promote, and retrieve MaxSight model artefacts.

Maintains a local registry JSON at runs/model_registry.json and optionally
mirrors to SageMaker Model Registry (Model Package Groups).

Registry workflow
-----------------
1. After training, call ``register_model()`` with the checkpoint path.
2. Review the entry; call ``promote_model()`` to move it to 'staging' or 'production'.
3. ``get_production_model()`` returns the active production checkpoint URI.

Usage
-----
from ml.infra.model_registry import ModelRegistry

registry = ModelRegistry()
registry.register_model(
    run_id="sm_20260301_120000",
    checkpoint_path=Path("model_output/best.pt"),
    metrics={"val_map": 0.52, "val_loss": 0.31},
    tier="T5_TEMPORAL",
)
registry.promote_model(run_id="sm_20260301_120000", stage="production")
entry = registry.get_production_model()
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY_PATH = REPO / "runs" / "model_registry.json"

STAGES = ("candidate", "staging", "production", "archived")


# ── Registry entry ────────────────────────────────────────────────────────────

@dataclass
class ModelEntry:
    run_id: str
    registered_at: str
    checkpoint_path: str
    s3_uri: str = ""
    tier: str = ""
    backbone: str = ""
    stage: str = "candidate"
    metrics: Dict[str, float] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "registered_at": self.registered_at,
            "checkpoint_path": self.checkpoint_path,
            "s3_uri": self.s3_uri,
            "tier": self.tier,
            "backbone": self.backbone,
            "stage": self.stage,
            "metrics": self.metrics,
            "tags": self.tags,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelEntry":
        return cls(
            run_id=d["run_id"],
            registered_at=d.get("registered_at", ""),
            checkpoint_path=d.get("checkpoint_path", ""),
            s3_uri=d.get("s3_uri", ""),
            tier=d.get("tier", ""),
            backbone=d.get("backbone", ""),
            stage=d.get("stage", "candidate"),
            metrics=d.get("metrics", {}),
            tags=d.get("tags", {}),
            notes=d.get("notes", ""),
        )


# ── Registry ──────────────────────────────────────────────────────────────────

class ModelRegistry:
    """Local + optional SageMaker Model Registry."""

    def __init__(
        self,
        registry_path: Path = DEFAULT_REGISTRY_PATH,
        *,
        s3_client=None,
        sm_model_package_group: Optional[str] = None,
        sm_config=None,
    ) -> None:
        self.registry_path = Path(registry_path)
        self._s3_client = s3_client
        self._sm_group = sm_model_package_group
        self._sm_cfg = sm_config
        self._entries: Dict[str, ModelEntry] = {}
        self._load()

    def _load(self) -> None:
        if self.registry_path.exists():
            data = json.loads(self.registry_path.read_text())
            self._entries = {k: ModelEntry.from_dict(v) for k, v in data.items()}

    def _save(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text(
            json.dumps({k: v.to_dict() for k, v in self._entries.items()}, indent=2),
            encoding="utf-8",
        )

    # ── CRUD ─────────────────────────────────────────────────────────────────

    def register_model(
        self,
        run_id: str,
        checkpoint_path: Path,
        *,
        metrics: Optional[Dict[str, float]] = None,
        tier: str = "",
        backbone: str = "",
        tags: Optional[Dict[str, str]] = None,
        notes: str = "",
        upload_to_s3: bool = False,
    ) -> ModelEntry:
        """Register a checkpoint as a new candidate model version."""
        s3_uri = ""
        if upload_to_s3 and self._s3_client and checkpoint_path.exists():
            s3_uri = self._s3_client.upload_checkpoint(
                checkpoint_path, run_id=run_id, tag="registered"
            )

        entry = ModelEntry(
            run_id=run_id,
            registered_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            checkpoint_path=str(checkpoint_path),
            s3_uri=s3_uri,
            tier=tier,
            backbone=backbone,
            stage="candidate",
            metrics=metrics or {},
            tags=tags or {},
            notes=notes,
        )
        self._entries[run_id] = entry
        self._save()

        if self._sm_group and s3_uri:
            self._register_sm_package(entry, s3_uri)

        logger.info("Registered model: %s [%s]", run_id, tier)
        return entry

    def promote_model(self, run_id: str, stage: str) -> ModelEntry:
        """Promote a model to 'staging' or 'production'.

        Demotes any existing production model to 'archived' first.
        """
        if stage not in STAGES:
            raise ValueError(f"Invalid stage {stage!r}. Choose from {STAGES}.")
        entry = self._get(run_id)
        if stage == "production":
            for e in self._entries.values():
                if e.stage == "production" and e.run_id != run_id:
                    e.stage = "archived"
        entry.stage = stage
        self._save()
        logger.info("Promoted %s → %s", run_id, stage)
        return entry

    def get_production_model(self) -> Optional[ModelEntry]:
        for e in reversed(list(self._entries.values())):
            if e.stage == "production":
                return e
        return None

    def get_stage_models(self, stage: str) -> List[ModelEntry]:
        return [e for e in self._entries.values() if e.stage == stage]

    def list_models(
        self,
        stage: Optional[str] = None,
        tier: Optional[str] = None,
        top_n: int = 20,
    ) -> List[ModelEntry]:
        entries = list(self._entries.values())
        if stage:
            entries = [e for e in entries if e.stage == stage]
        if tier:
            entries = [e for e in entries if e.tier == tier]
        return entries[-top_n:]

    def compare_models(
        self,
        metric: str = "val_map",
        stage: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return models sorted by a metric (descending)."""
        entries = self.list_models(stage=stage)
        rows = [
            {
                "run_id": e.run_id,
                "tier": e.tier,
                "stage": e.stage,
                "registered_at": e.registered_at,
                metric: e.metrics.get(metric),
            }
            for e in entries
        ]
        rows.sort(key=lambda r: (r[metric] or 0.0), reverse=True)
        return rows

    def tag_model(self, run_id: str, **tags: str) -> None:
        self._get(run_id).tags.update(tags)
        self._save()

    def _get(self, run_id: str) -> ModelEntry:
        if run_id not in self._entries:
            raise KeyError(f"run_id {run_id!r} not in registry")
        return self._entries[run_id]

    # ── SageMaker Model Package integration ───────────────────────────────────

    def _register_sm_package(self, entry: ModelEntry, model_data_s3: str) -> None:
        # Propagate exceptions when a package group is configured — callers depend on
        # the SageMaker ARN being present for the registry gate to work correctly.
        import boto3
        sm = boto3.client("sagemaker", region_name=getattr(self._sm_cfg, "region", "us-east-1"))
        sm.create_model_package(
            ModelPackageGroupName=self._sm_group,
            ModelPackageDescription=f"MaxSight {entry.tier} {entry.run_id}",
            InferenceSpecification={
                "Containers": [{
                    "Image": getattr(self._sm_cfg, "inference_image", ""),
                    "ModelDataUrl": model_data_s3,
                }],
                "SupportedContentTypes": ["application/json"],
                "SupportedResponseMIMETypes": ["application/json"],
            },
            ModelApprovalStatus="PendingManualApproval",
        )
        logger.info("Registered SageMaker Model Package: %s / %s", self._sm_group, entry.run_id)
