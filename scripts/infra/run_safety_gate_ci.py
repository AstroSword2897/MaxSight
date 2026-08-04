#!/usr/bin/env python3
"""Thin safety-gate CI wrapper. Emits frozen manifest shape from ml.evaluation.safety_gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ml.evaluation.safety_gates import (  # noqa: E402
    build_certification_manifest,
    evaluate_condition_platform_cell,
    format_cell_status_line,
)
from ml.runtime_constants import CONDITION_MODE_IDS  # noqa: E402


def _artifact_hash(path: Path | None) -> str:
    if path is None or not path.is_file():
        return "missing-artifact"
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run fail-closed safety-gate certification")
    parser.add_argument("--artifact", type=Path, default=None)
    parser.add_argument("--platform", default="torch_ref")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--hazard-gt",
        action="store_true",
        help="Declare hazard ground truth available (default: blocked SG-01/02)",
    )
    parser.add_argument(
        "--tools-missing",
        action="store_true",
        help="Mark platform tools missing → skipped_tools_missing [SKIP]",
    )
    parser.add_argument(
        "--force-xfail",
        action="store_true",
        help="Force xfail_known_issue [XFAIL] for all cells (test harness)",
    )
    args = parser.parse_args(argv)

    modes = [m for m in CONDITION_MODE_IDS if m != "none"]
    cells = []
    for mode in modes:
        force = "xfail_known_issue" if args.force_xfail else None
        cell = evaluate_condition_platform_cell(
            condition_mode=mode,
            platform=args.platform,
            hazard_ground_truth_available=bool(args.hazard_gt),
            tools_available=not bool(args.tools_missing),
            force_status=force,
        )
        print(format_cell_status_line(cell))
        cells.append(cell)

    manifest = build_certification_manifest(
        artifact_hash=_artifact_hash(args.artifact),
        platform=args.platform,
        cells=cells,
    )
    summary = manifest["summary"]
    print(
        "summary "
        f"passed={summary['passed']} failed={summary['failed']} "
        f"blocked={summary['blocked']} skipped={summary['skipped']} "
        f"xfailed={summary['xfailed']} all_passed={manifest['all_passed']}"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0 if manifest["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
