"""Production video ingest pipeline: batch + stream paths with corruption isolation."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Generator, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from ml.data.video_ingest_validator import VideoIngestReport, validate_video_file

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = [".mp4", ".mov", ".mkv", ".webm"]


@dataclass
class IngestStats:
    """Running totals for a pipeline run."""

    total: int = 0
    accepted: int = 0
    rejected: int = 0
    errors: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0

    @property
    def rejection_rate(self) -> float:
        return self.rejected / max(1, self.total)

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "rejection_rate": round(self.rejection_rate, 4),
            "errors": self.errors[:50],
            "elapsed_s": round(self.elapsed_s, 3),
        }


class VideoIngestPipeline:
    """Validate and route video files; never silently discard corrupt inputs."""

    def __init__(
        self,
        *,
        min_bytes: int = 1024,
        allowed_suffixes: list[str] | None = None,
        on_rejected: Callable[[VideoIngestReport], None] | None = None,
    ) -> None:
        self.min_bytes = min_bytes
        self.allowed_suffixes = allowed_suffixes or SUPPORTED_SUFFIXES
        self.on_rejected = on_rejected or self._default_rejected

    @staticmethod
    def _default_rejected(report: VideoIngestReport) -> None:
        logger.warning("ingest_rejected path=%s errors=%s", report.path, report.errors)

    def process_batch(self, paths: Iterable[Path]) -> IngestStats:
        """Validate a batch of video files; return statistics."""
        stats = IngestStats()
        t0 = time.perf_counter()
        for path in paths:
            report = validate_video_file(
                path, min_bytes=self.min_bytes, allowed_suffixes=self.allowed_suffixes
            )
            stats.total += 1
            if report.valid:
                stats.accepted += 1
            else:
                stats.rejected += 1
                stats.errors.extend(f"{path.name}:{e}" for e in report.errors)
                self.on_rejected(report)
        stats.elapsed_s = time.perf_counter() - t0
        logger.info("ingest_batch stats=%s", stats.to_dict())
        return stats

    def stream_valid(self, paths: Iterable[Path]) -> Generator[Path, None, None]:
        """Yield only valid paths; log and skip corrupt ones."""
        for path in paths:
            report = validate_video_file(
                path, min_bytes=self.min_bytes, allowed_suffixes=self.allowed_suffixes
            )
            if report.valid:
                yield path
            else:
                self.on_rejected(report)


def discover_videos(root: Path, recursive: bool = True) -> list[Path]:
    """Return sorted list of video files under root."""
    pattern = "**/*" if recursive else "*"
    found = []
    for suffix in SUPPORTED_SUFFIXES:
        found.extend(root.glob(pattern + suffix))
    return sorted(set(found))
