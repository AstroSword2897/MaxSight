"""Model artifact storage with staging-only writes by default (ACTIVE denied)."""

from __future__ import annotations

from pathlib import Path


class ActivePointerWriteDenied(PermissionError):
    """Raised when staging/download code attempts to write ACTIVE_MODEL_PTR."""


class ModelArtifactStore:
    """Filesystem-backed store. Staging cannot mutate the active pointer."""

    ACTIVE_NAME = "ACTIVE_MODEL_PTR"
    STAGING_DIRNAME = "staging"

    def __init__(self, root: str | Path, *, allow_active_writes: bool = False) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.staging_dir = self.root / self.STAGING_DIRNAME
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.allow_active_writes = allow_active_writes
        self.active_path = self.root / self.ACTIVE_NAME

    def write_staging(self, name: str, data: bytes) -> Path:
        if name == self.ACTIVE_NAME or Path(name).name == self.ACTIVE_NAME:
            raise ActivePointerWriteDenied("staging cannot write ACTIVE_MODEL_PTR")
        path = self.staging_dir / name
        path.write_bytes(data)
        return path

    def write_active_pointer(self, target: str | Path) -> Path:
        if not self.allow_active_writes:
            raise ActivePointerWriteDenied(
                "ACTIVE_MODEL_PTR writes denied (enable only via activation factory)"
            )
        self.active_path.write_text(str(target) + "\n", encoding="utf-8")
        return self.active_path

    def read_active_pointer(self) -> str | None:
        if not self.active_path.is_file():
            return None
        return self.active_path.read_text(encoding="utf-8").strip() or None


def staging_store(root: str | Path) -> ModelArtifactStore:
    return ModelArtifactStore(root, allow_active_writes=False)


def activation_store(root: str | Path) -> ModelArtifactStore:
    """Sole First Wave path allowed to mutate ACTIVE_MODEL_PTR."""
    return ModelArtifactStore(root, allow_active_writes=True)
