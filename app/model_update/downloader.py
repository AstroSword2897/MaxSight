"""Local 'release bucket' downloader for OTA staging tests."""

from __future__ import annotations

from pathlib import Path

from app.model_update.storage import ModelArtifactStore


class LocalReleaseDownloader:
    """Copy artifacts from a local release directory into staging (never ACTIVE)."""

    def __init__(self, store: ModelArtifactStore, release_root: Path | str) -> None:
        self.store = store
        self.release_root = Path(release_root)

    def download(self, relative_name: str) -> Path:
        src = self.release_root / relative_name
        if not src.is_file():
            raise FileNotFoundError(src)
        return self.store.write_staging(Path(relative_name).name, src.read_bytes())
