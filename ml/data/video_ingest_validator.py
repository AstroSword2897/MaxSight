"""Video ingestion validation with corruption detection."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass
class VideoIngestReport:
    """Result of video ingest validation."""

    path: str
    valid: bool
    frame_count: int
    checksum: str
    errors: list[str]

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "valid": self.valid,
            "frame_count": self.frame_count,
            "checksum": self.checksum,
            "errors": self.errors,
        }


def _file_checksum(path: Path, chunk_size: int = 65536) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def validate_video_file(
    path: Path,
    *,
    min_bytes: int = 256,
    allowed_suffixes: list[str] | None = None,
) -> VideoIngestReport:
    """Validate a video file exists, has content, and matches allowed formats."""
    suffixes = allowed_suffixes or [".mp4", ".mov", ".mkv", ".webm"]
    errors: list[str] = []
    if not path.exists():
        return VideoIngestReport(str(path), False, 0, "", ["file_not_found"])
    if path.stat().st_size < min_bytes:
        errors.append("file_too_small")
    if path.suffix.lower() not in suffixes:
        errors.append(f"unsupported_format:{path.suffix}")
    checksum = _file_checksum(path)
    frame_count = 0
    try:
        import cv2  # type: ignore[import-untyped]

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            errors.append("opencv_open_failed")
        else:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if frame_count <= 0:
                errors.append("zero_frames")
            cap.release()
    except ImportError:
        # Headless CI may lack cv2; size/checksum checks still apply.
        if not errors:
            frame_count = -1
    except Exception as exc:
        errors.append(f"frame_read_error:{exc}")
    return VideoIngestReport(
        path=str(path),
        valid=len(errors) == 0,
        frame_count=frame_count,
        checksum=checksum,
        errors=errors,
    )
