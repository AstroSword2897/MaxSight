"""OTA staging: validate then stage — never auto-promote."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from app.model_update.downloader import LocalReleaseDownloader
from app.model_update.storage import ActivePointerWriteDenied, ModelArtifactStore, staging_store


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage_candidate(
    *,
    store: ModelArtifactStore,
    artifact_path: Path,
    manifest: dict[str, Any],
    require_all_passed: bool = True,
) -> Path:
    """Copy a validated candidate into staging. Raises if manifest not all_passed."""
    if require_all_passed and not manifest.get("all_passed"):
        raise ValueError("refuse to stage: certification manifest all_passed is not True")
    if store.allow_active_writes:
        raise ActivePointerWriteDenied("use staging_store() for OTA staging")
    name = artifact_path.name
    staged = store.write_staging(name, artifact_path.read_bytes())
    meta = {
        "artifact_hash": _sha256(staged),
        "manifest": manifest,
        "staged_name": name,
    }
    store.write_staging(f"{name}.stage.json", json.dumps(meta, indent=2).encode("utf-8"))
    return staged


def ota_download_and_stage(
    *,
    root: Path | str,
    release_root: Path | str,
    relative_name: str,
    manifest: dict[str, Any],
) -> Path:
    store = staging_store(root)
    downloader = LocalReleaseDownloader(store, release_root)
    downloaded = downloader.download(relative_name)
    return stage_candidate(store=store, artifact_path=downloaded, manifest=manifest)
