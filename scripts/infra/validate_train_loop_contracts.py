#!/usr/bin/env python3
"""Validate ProductionTrainLoop observability contracts used by CI and CloudWatch."""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.infra.sagemaker_utils import TRAINING_METRIC_DEFINITIONS  # noqa: E402
from ml.training.observability import (  # noqa: E402
    EVENT_LOG_PREFIX,
    HEALTH_SUMMARY_LOG_PREFIX,
    HEALTH_SUMMARY_REQUIRED_FIELDS,
    STRUCTURED_EVENT_SCHEMAS,
    parse_health_summary_line,
)


def _errors() -> list[str]:
    errors: list[str] = []
    source = Path(REPO_ROOT / "ml/training/train_loop.py").read_text(encoding="utf-8")

    if "skipped_batches" not in source:
        errors.append("train_loop.py must track skipped_batches per epoch.")
    if HEALTH_SUMMARY_LOG_PREFIX not in source:
        errors.append(f"train_loop.py must emit '{HEALTH_SUMMARY_LOG_PREFIX}' logs.")
    if "max_skipped_batch_ratio" not in source:
        errors.append("train_loop.py must define/use max_skipped_batch_ratio.")

    # Source-check ProductionTrainLoop.__init__ so contracts CI need not import torch.
    init_match = re.search(
        r"class ProductionTrainLoop\b[\s\S]*?def __init__\((.*?)\)\s*->",
        source,
        re.DOTALL,
    )
    if init_match is None:
        errors.append("Could not locate ProductionTrainLoop.__init__ signature in train_loop.py.")
    else:
        init_sig = init_match.group(1)
        if "max_skipped_batch_ratio" not in init_sig:
            errors.append("ProductionTrainLoop.__init__ missing max_skipped_batch_ratio parameter.")
        if "DEFAULT_MAX_SKIPPED_BATCH_RATIO" not in init_sig:
            errors.append(
                "ProductionTrainLoop max_skipped_batch_ratio default must match observability constant."
            )

    sample_with_val = (
        "health_summary epoch=1 processed_batches=10 skipped_batches=1 skip_ratio=9.09% "
        "train_loss=1.2345 val_loss=2.3456 val_map=0.4567 new_best=True lr=1.000000e-03"
    )
    sample_no_val = (
        "health_summary epoch=2 processed_batches=8 skipped_batches=0 skip_ratio=0.00% "
        "train_loss=0.5000 new_best=False lr=5.000000e-04"
    )
    for sample in (sample_with_val, sample_no_val):
        try:
            parsed = parse_health_summary_line(sample)
        except ValueError as exc:
            errors.append(f"health_summary parser failed: {exc}")
            continue
        for field in HEALTH_SUMMARY_REQUIRED_FIELDS:
            if not hasattr(parsed, field.replace("-", "_")) and field not in {
                "epoch",
                "processed_batches",
                "skipped_batches",
                "skip_ratio",
                "train_loss",
                "new_best",
                "lr",
            }:
                continue

    health_metric_names = {
        m["Name"] for m in TRAINING_METRIC_DEFINITIONS if m["Name"].startswith("train:")
    }
    expected = {
        "train:loss",
        "train:processed_batches",
        "train:skipped_batches",
        "train:skip_ratio_pct",
    }
    missing = expected - health_metric_names
    if missing:
        errors.append(f"TRAINING_METRIC_DEFINITIONS missing health metrics: {sorted(missing)}")

    for mdef in TRAINING_METRIC_DEFINITIONS:
        if not mdef["Name"].startswith("train:") and "health_summary" in mdef["Regex"]:
            continue
        if mdef["Name"].startswith("train:") and "health_summary" in mdef.get("Regex", ""):
            if not re.search(mdef["Regex"], sample_with_val):
                errors.append(f"Regex does not match health_summary sample for {mdef['Name']}")

    observability_source = Path(REPO_ROOT / "ml/training/observability.py").read_text(
        encoding="utf-8"
    )
    if EVENT_LOG_PREFIX not in observability_source:
        errors.append("observability.py must define structured event log prefix.")
    if "def emit_event" not in observability_source:
        errors.append("observability.py must expose emit_event().")
    required_events = {
        "training.health_summary",
        "therapy.suppressed",
        "rag.degraded",
        "rag.failure",
        "rag.alert",
        "runtime.tier_resolved",
    }
    missing_events = required_events - set(STRUCTURED_EVENT_SCHEMAS.keys())
    if missing_events:
        errors.append(f"STRUCTURED_EVENT_SCHEMAS missing: {sorted(missing_events)}")

    return errors


def main() -> int:
    errors = _errors()
    if errors:
        for line in errors:
            print(line, file=sys.stderr)
        return 1
    print("OK: train loop observability contracts validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
