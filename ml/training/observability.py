"""Training loop observability contracts shared by runtime logs and CI checks."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

# Default skipped-batch abort threshold used by ProductionTrainLoop.
DEFAULT_MAX_SKIPPED_BATCH_RATIO = 0.1

# Prefix for structured epoch health logs emitted at end of each epoch.
HEALTH_SUMMARY_LOG_PREFIX = "health_summary"
EVENT_LOG_PREFIX = "event="

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

STRUCTURED_EVENT_SCHEMAS: dict[str, frozenset[str]] = {
    "training.health_summary": frozenset(HEALTH_SUMMARY_REQUIRED_FIELDS),
    "therapy.suppressed": frozenset({"reason", "count", "module", "function"}),
    "rag.degraded": frozenset({"guard_reason", "latency_ms"}),
    "runtime.tier_resolved": frozenset({"tier", "enable_rag", "enable_therapy"}),
}

# Regex used by CloudWatch metric extraction and CI contract tests.
HEALTH_SUMMARY_LOG_REGEX = (
    r"health_summary epoch=(?P<epoch>\d+) processed_batches=(?P<processed_batches>\d+) "
    r"skipped_batches=(?P<skipped_batches>\d+) skip_ratio=(?P<skip_ratio>[0-9.]+)% "
    r"train_loss=(?P<train_loss>[0-9.eE+-]+)(?: val_loss=(?P<val_loss>[0-9.eE+-]+) "
    r"val_map=(?P<val_map>[0-9.eE+-]+))? new_best=(?P<new_best>True|False) lr=(?P<lr>[0-9.eE+-]+)"
)

HEALTH_SUMMARY_PATTERN = re.compile(HEALTH_SUMMARY_LOG_REGEX)


@dataclass
class StructuredEvent:
    """Structured JSON log event validated against a named schema."""

    name: str
    fields: dict[str, Any]

    def validate(self) -> None:
        """Raise ValueError when required fields for the schema are missing."""
        required = STRUCTURED_EVENT_SCHEMAS.get(self.name)
        if required is None:
            raise ValueError(f"unknown structured event schema: {self.name}")
        missing = required - set(self.fields.keys())
        if missing:
            raise ValueError(f"event {self.name} missing fields: {sorted(missing)}")

    def to_log_line(self) -> str:
        self.validate()
        payload = {"event": self.name, **self.fields}
        return f"{EVENT_LOG_PREFIX}{json.dumps(payload, sort_keys=True, default=str)}"


def emit_event(name: str, **fields: Any) -> None:
    """Emit a structured event as a single JSON log line."""
    event = StructuredEvent(name=name, fields=fields)
    logging.getLogger(__name__).info(event.to_log_line())


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
        raise RuntimeError(f"Skipped batch ratio {ratio:.2%} exceeded threshold {max_ratio:.2%}.")


def cloudwatch_health_metric_definitions() -> tuple[dict[str, str], ...]:
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
