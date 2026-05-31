#!/usr/bin/env python3
"""Run baseline quality audit commands adapted for this repository layout."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# Tier 1 production core — strict gates apply here only.
TARGETS = ["ml/therapy", "ml/runtime", "app/personal_mode.py"]
MYPY_TARGETS = ["ml/therapy", "ml/runtime", "app/personal_mode.py"]
BASELINE_PATH = REPO / "docs" / "quality" / "baseline.json"
BASELINE_CC_PATH = REPO / "docs" / "quality" / "baseline_cc.txt"
BASELINE_RUFF_PATH = REPO / "docs" / "quality" / "baseline_ruff.txt"
RUFF_JSON_PATH = REPO / "ruff_baseline.json"
MYPY_BASELINE_PATH = REPO / "docs" / "quality" / "mypy_baseline.txt"


def _run(cmd: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=REPO, capture_output=capture, text=True)


def _parse_radon_cc() -> dict:
    result = _run(
        ["radon", "cc", *TARGETS, "-j"],
        capture=True,
    )
    d_rated: list[str] = []
    c_rated: list[str] = []
    complexities: list[float] = []
    if result.stdout.strip():
        payload = json.loads(result.stdout)
        for path, blocks in payload.items():
            rel = str(Path(path).relative_to(REPO)) if Path(path).is_absolute() else path
            for block in blocks:
                rank = block.get("rank", "")
                name = block.get("name", "")
                label = f"{rel}:{name}"
                complexity = float(block.get("complexity", 0))
                complexities.append(complexity)
                if rank == "D":
                    d_rated.append(label)
                elif rank == "C":
                    c_rated.append(label)
    avg = sum(complexities) / len(complexities) if complexities else 0.0
    return {
        "d_rated": sorted(d_rated),
        "c_rated": sorted(c_rated),
        "avg_complexity": round(avg, 2),
    }


def _parse_ruff() -> dict:
    result = _run(
        ["ruff", "check", *TARGETS, "--output-format=json"],
        capture=True,
    )
    text_result = _run(["ruff", "check", *TARGETS], capture=True)
    violations: list[dict] = []
    if result.stdout.strip():
        violations = json.loads(result.stdout)
    RUFF_JSON_PATH.write_text(json.dumps(violations, indent=2), encoding="utf-8")
    BASELINE_RUFF_PATH.write_text(text_result.stdout or text_result.stderr, encoding="utf-8")
    by_code = Counter(v.get("code", "unknown") for v in violations)
    return {"total": len(violations), "by_code": dict(by_code.most_common())}


def _parse_mypy() -> dict:
    result = _run(
        [
            "mypy",
            *MYPY_TARGETS,
            "--follow-imports=silent",
            "--ignore-missing-imports",
        ],
        capture=True,
    )
    MYPY_BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    output = (result.stdout or "") + (result.stderr or "")
    MYPY_BASELINE_PATH.write_text(output, encoding="utf-8")
    error_lines = [line for line in output.splitlines() if ": error:" in line]
    codes = Counter()
    for line in error_lines:
        match = re.search(r": error: (.+?)  \[(.+?)\]", line)
        if match:
            codes[match.group(2)] += 1
    return {
        "therapy_runtime_errors": len(error_lines),
        "follow_imports": "silent",
        "top_error_codes": dict(codes.most_common(10)),
        "exit_code": result.returncode,
    }


def _run_xenon() -> dict:
    result = _run(
        [
            "xenon",
            *TARGETS,
            "--max-absolute",
            "B",
            "--max-average",
            "A",
            "--max-modules",
            "B",
        ],
        capture=True,
    )
    return {"passed": result.returncode == 0, "exit_code": result.returncode}


def _write_radon_cc_snapshot() -> None:
    result = _run(["radon", "cc", *TARGETS, "-s"], capture=True)
    BASELINE_CC_PATH.write_text(result.stdout or result.stderr, encoding="utf-8")


def _check_regression(snapshot: dict, prior: dict | None) -> list[str]:
    if prior is None:
        return []
    errors: list[str] = []
    new_d = set(snapshot["radon"]["d_rated"]) - set(prior.get("radon", {}).get("d_rated", []))
    if new_d:
        errors.append(f"New D-rated blocks: {sorted(new_d)}")
    prior_ruff = prior.get("ruff", {}).get("total", 0)
    current_ruff = snapshot["ruff"]["total"]
    if current_ruff > prior_ruff:
        errors.append(f"Ruff violations increased: {prior_ruff} -> {current_ruff}")
    return errors


def main() -> int:
    REPO.joinpath("docs/quality").mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Quality baseline audit")
    print("=" * 60)

    mi_result = _run(["radon", "mi", *TARGETS, "-nb", "--min", "B"])
    print(mi_result.stdout or mi_result.stderr)

    cc_summary = _parse_radon_cc()
    _write_radon_cc_snapshot()
    print(
        f"radon cc: D={len(cc_summary['d_rated'])} C={len(cc_summary['c_rated'])} avg={cc_summary['avg_complexity']}"
    )
    print(f"  -> {BASELINE_CC_PATH.relative_to(REPO)}")

    ruff_summary = _parse_ruff()
    print(f"ruff: {ruff_summary['total']} violations -> {RUFF_JSON_PATH.name}")
    print(f"  -> {BASELINE_RUFF_PATH.relative_to(REPO)}")

    mypy_summary = _parse_mypy()
    print(f"mypy: {mypy_summary['therapy_runtime_errors']} errors -> {MYPY_BASELINE_PATH}")

    xenon_summary = _run_xenon()
    print(f"xenon: {'PASS' if xenon_summary['passed'] else 'FAIL'}")

    prior: dict | None = None
    if BASELINE_PATH.exists():
        prior = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))

    snapshot = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "targets": TARGETS,
        "radon": cc_summary,
        "ruff": ruff_summary,
        "mypy": mypy_summary,
        "xenon": xenon_summary,
    }
    BASELINE_PATH.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    print(f"Wrote {BASELINE_PATH.relative_to(REPO)}")

    regressions = _check_regression(snapshot, prior)
    exit_code = 0
    if cc_summary["d_rated"]:
        print(f"FAIL: D-rated blocks present: {cc_summary['d_rated']}", file=sys.stderr)
        exit_code = 1
    for msg in regressions:
        print(f"FAIL: {msg}", file=sys.stderr)
        exit_code = 1
    if exit_code == 0:
        print("Baseline audit PASSED")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
