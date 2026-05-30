"""Training loop observability contracts shared by runtime logs and CI checks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Tuple

# Default skipped-batch abort threshold used by ProductionTrainLoop.
DEFAULT_MAX_SKIPPED_BATCH_RATIO = 0.1

# Prefix for structured epoch health logs emitted at end of each epoch.
HEALTH_SUMMARY_LOG_PREFIX = "health_summary"

# Required keys in health_summary log lines (key=value tokens).
HEALTH_SUMMARY_REQUIRED_FIELDS = (
    "epoch",
    "processed_batches",
    "skipped_batches",
    "skip_ratio",
    "train_loss",
    "new_best",
    "lr",
)

# Regex used by CloudWatch metric extraction and CI contract tests.
HEALTH_SUMMARY_LOG_REGEX = (
    r"health_summary epoch=(?P<epoch>\d+) processed_batches=(?P<processed_batches>\d+) "
    r"skipped_batches=(?P<skipped_batches>\d+) skip_ratio=(?P<skip_ratio>[0-9.]+)% "
    r"train_loss=(?P<train_loss>[0-9.eE+-]+)(?: val_loss=(?P<val_loss>[0-9.eE+-]+) "
    r"val_map=(?P<val_map>[0-9.eE+-]+))? new_best=(?P<new_best>True|False) lr=(?P<lr>[0-9.eE+-]+)"
)

HEALTH_SUMMARY_PATTERN = re.compile(HEALTH_SUMMARY_LOG_REGEX)


@dataclass(frozen=True)
class TrainingHealthSummary:
    """Parsed epoch health summary from a training log line."""

    epoch: int
    processed_batches: int
    skipped_batches: int
    skip_ratio_percent: float
    train_loss: float
    new_best: bool
    lr: float
    val_loss: float | None = None
    val_map: float | None = None


def parse_health_summary_line(line: str) -> TrainingHealthSummary:
    """Parse a health_summary log line into structured fields.

    Raises:
        ValueError: When the line does not match the contract regex.
    """
    match = HEALTH_SUMMARY_PATTERN.search(line)
    if match is None:
        raise ValueError(f"health_summary line does not match contract: {line!r}")
    groups = match.groupdict()
    return TrainingHealthSummary(
        epoch=int(groups["epoch"]),
        processed_batches=int(groups["processed_batches"]),
        skipped_batches=int(groups["skipped_batches"]),
        skip_ratio_percent=float(groups["skip_ratio"]),
        train_loss=float(groups["train_loss"]),
        new_best=groups["new_best"] == "True",
        lr=float(groups["lr"]),
        val_loss=float(groups["val_loss"]) if groups.get("val_loss") else None,
        val_map=float(groups["val_map"]) if groups.get("val_map") else None,
    )


def validate_skipped_batch_ratio(
    skipped_batches: int,
    total_batches: int,
    *,
    max_ratio: float = DEFAULT_MAX_SKIPPED_BATCH_RATIO,
) -> None:
    """Raise when skipped batch ratio exceeds the configured threshold."""
    if total_batches <= 0:
        return
    ratio = skipped_batches / total_batches
    if ratio > max_ratio:
        raise RuntimeError(
            f"Skipped batch ratio {ratio:.2%} exceeded threshold {max_ratio:.2%}."
        )


def cloudwatch_health_metric_definitions() -> Tuple[Dict[str, str], ...]:
    """CloudWatch metric definitions derived from health_summary logs."""
    return (
        {
            "Name": "train:processed_batches",
            "Regex": r"health_summary epoch=\d+ processed_batches=(\d+)",
        },
        {
            "Name": "train:skipped_batches",
            "Regex": r"health_summary epoch=\d+ processed_batches=\d+ skipped_batches=(\d+)",
        },
        {
            "Name": "train:skip_ratio_pct",
            "Regex": r"health_summary epoch=\d+ processed_batches=\d+ skipped_batches=\d+ skip_ratio=([0-9.]+)%",
        },
    )
