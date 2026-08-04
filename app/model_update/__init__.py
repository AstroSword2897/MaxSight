"""OTA model update package (app layer; isolated from Stage A)."""

from app.model_update.activation import promote_or_rollback, rollback_to_previous
from app.model_update.staging import ota_download_and_stage, stage_candidate
from app.model_update.storage import (
    ActivePointerWriteDenied,
    ModelArtifactStore,
    activation_store,
    staging_store,
)

__all__ = [
    "ActivePointerWriteDenied",
    "ModelArtifactStore",
    "activation_store",
    "ota_download_and_stage",
    "promote_or_rollback",
    "rollback_to_previous",
    "stage_candidate",
    "staging_store",
]
