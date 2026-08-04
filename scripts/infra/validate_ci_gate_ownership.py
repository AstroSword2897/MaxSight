#!/usr/bin/env python3
"""Fail-closed ownership checks for CI vs Quality workflow boundaries.

Parses workflow YAML structurally (not regex-over-text) and inspects executed
command surfaces only: job steps' `run` fields and reusable-workflow `with.commands`.
"""

from __future__ import annotations

import shlex
import sys
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
except ImportError:  # pragma: no cover
    print("FAIL: PyYAML is required (pip install pyyaml)", file=sys.stderr)
    raise SystemExit(2)

REPO = Path(__file__).resolve().parents[2]
CI = REPO / ".github/workflows/ci.yml"
QUALITY = REPO / ".github/workflows/quality.yml"

# Status enums used by certification manifests (must stay aligned with ml.evaluation.safety_gates).
ALLOWED_CELL_STATUSES = frozenset(
    {
        "passed",
        "failed",
        "blocked_missing_hazard_labels",
        "skipped_tools_missing",
        "xfail_known_issue",
    }
)

# Phone First Wave packages owned by ci.yml phone-app-layer (not quality ruff).
# app/ui and app/personal_mode.py may be linted by quality when intentional.
PHONE_RUFF_FORBIDDEN = ("app/connectivity", "app/stage_b", "app/model_update")

ISOLATION_TOKEN = "validate_stage_a_isolation.py"
QUALITY_AUDIT_TOKEN = "run_quality_audit.py"
CONDITION_CONTRACT = "test_condition_tensor_contract.py"


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)


def _load_workflow(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not parse to a mapping")
    return data


def _iter_executable_commands(workflow: dict[str, Any]) -> Iterable[tuple[str, str, str]]:
    """Yield (job_id, step_name, command_text) for executed surfaces only."""
    jobs = workflow.get("jobs") or {}
    if not isinstance(jobs, dict):
        return
    for job_id, job in jobs.items():
        if not isinstance(job, dict):
            continue
        # Reusable workflow_call: commands live under `with.commands`.
        with_block = job.get("with")
        if isinstance(with_block, dict):
            commands = with_block.get("commands")
            if isinstance(commands, str) and commands.strip():
                yield job_id, "with.commands", commands
        steps = job.get("steps") or []
        if not isinstance(steps, list):
            continue
        for idx, step in enumerate(steps):
            if not isinstance(step, dict):
                continue
            run = step.get("run")
            if not isinstance(run, str) or not run.strip():
                continue
            step_name = str(step.get("name") or f"steps[{idx}]")
            yield job_id, step_name, run


def _tokenize(command: str) -> list[str]:
    """Normalize a shell command block into comparable tokens (best-effort)."""
    tokens: list[str] = []
    for line in command.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # Drop line-continuation markers so shlex sees logical lines.
        if stripped.endswith("\\"):
            stripped = stripped[:-1].rstrip()
        try:
            tokens.extend(shlex.split(stripped, posix=True))
        except ValueError:
            tokens.extend(stripped.split())
    return tokens


def _command_invokes(command: str, filename: str) -> bool:
    """True when filename appears as an executed path token (not only in a comment)."""
    tokens = _tokenize(command)
    for tok in tokens:
        base = Path(tok).name
        if tok == filename or tok.endswith("/" + filename) or base == filename:
            return True
    return False


def _check_stage_a_isolation(ci: dict[str, Any], quality: dict[str, Any], errors: list[str]) -> None:
    quality_hits = [
        (job, step)
        for job, step, cmd in _iter_executable_commands(quality)
        if _command_invokes(cmd, ISOLATION_TOKEN) or _command_invokes(cmd, "validate_stage_a_isolation")
    ]
    for job, step in quality_hits:
        errors.append(
            f"quality.yml job '{job}' step '{step}' must not invoke {ISOLATION_TOKEN} "
            "(Stage A isolation is owned by ci.yml stage-a-contracts only)."
        )

    ci_jobs = set((ci.get("jobs") or {}).keys())
    if "stage-a-contracts" not in ci_jobs:
        errors.append("ci.yml must define job 'stage-a-contracts'.")
    stage_cmds = [
        cmd
        for job, _step, cmd in _iter_executable_commands(ci)
        if job == "stage-a-contracts"
    ]
    if not any(_command_invokes(cmd, ISOLATION_TOKEN) for cmd in stage_cmds):
        errors.append(
            f"ci.yml job 'stage-a-contracts' must invoke {ISOLATION_TOKEN} "
            "(with.commands / run steps)."
        )


def _check_quality_audit(ci: dict[str, Any], quality: dict[str, Any], errors: list[str]) -> None:
    for job, step, cmd in _iter_executable_commands(ci):
        if _command_invokes(cmd, QUALITY_AUDIT_TOKEN):
            errors.append(
                f"ci.yml job '{job}' step '{step}' must not invoke {QUALITY_AUDIT_TOKEN} "
                "(quality baseline drift is owned by quality.yml quality-drift only)."
            )

    quality_jobs = set((quality.get("jobs") or {}).keys())
    if "quality-drift" not in quality_jobs:
        errors.append("quality.yml must define job 'quality-drift'.")
    drift_cmds = [
        cmd
        for job, _step, cmd in _iter_executable_commands(quality)
        if job == "quality-drift"
    ]
    if not any(_command_invokes(cmd, QUALITY_AUDIT_TOKEN) for cmd in drift_cmds):
        errors.append(
            f"quality.yml job 'quality-drift' must invoke {QUALITY_AUDIT_TOKEN}."
        )


def _check_ruff_scope(quality: dict[str, Any], errors: list[str]) -> None:
    jobs = quality.get("jobs") or {}
    ruff_job = jobs.get("ruff-tier1")
    if not isinstance(ruff_job, dict):
        errors.append("quality.yml must define job 'ruff-tier1'.")
        return
    for step in ruff_job.get("steps") or []:
        if not isinstance(step, dict):
            continue
        run = step.get("run")
        if not isinstance(run, str):
            continue
        step_name = str(step.get("name") or "run")
        tokens = _tokenize(run)
        flat_tokens = set(tokens)
        for banned in PHONE_RUFF_FORBIDDEN:
            if banned in flat_tokens or any(t.startswith(banned + "/") for t in tokens):
                errors.append(
                    f"quality.yml job 'ruff-tier1' step '{step_name}' targets '{banned}' "
                    "(phone app layer is owned by ci.yml phone-app-layer)."
                )
        # Stage A package may only appear as an exclusion target.
        if "ml/runtime/stage_a" in flat_tokens or any(
            t.startswith("ml/runtime/stage_a/") for t in tokens
        ):
            if "--extend-exclude" not in flat_tokens and "--exclude" not in flat_tokens:
                errors.append(
                    f"quality.yml job 'ruff-tier1' step '{step_name}' targets "
                    "ml/runtime/stage_a without an exclude flag "
                    "(Stage A boundary owned by ci.yml)."
                )


def _check_condition_tensor(ci: dict[str, Any], errors: list[str]) -> None:
    drift_hits = [
        (job, step)
        for job, step, cmd in _iter_executable_commands(ci)
        if job == "drift-checks" and _command_invokes(cmd, CONDITION_CONTRACT)
    ]
    for job, step in drift_hits:
        errors.append(
            f"ci.yml job '{job}' step '{step}' must not run {CONDITION_CONTRACT} "
            "(owned by torch-condition-tensor)."
        )

    torch_cmds = [
        (job, step, cmd)
        for job, step, cmd in _iter_executable_commands(ci)
        if job == "torch-condition-tensor"
    ]
    if not torch_cmds:
        errors.append("ci.yml must define executable commands for job 'torch-condition-tensor'.")
    elif not any(_command_invokes(cmd, CONDITION_CONTRACT) for _, _, cmd in torch_cmds):
        job, step, _ = torch_cmds[0]
        errors.append(
            f"ci.yml job '{job}' step '{step}' must run {CONDITION_CONTRACT}."
        )


def main() -> int:
    errors: list[str] = []
    try:
        ci = _load_workflow(CI)
        quality = _load_workflow(QUALITY)
    except Exception as exc:  # noqa: BLE001 — fail-closed with actionable message
        _fail(f"failed to parse workflow YAML: {exc}")
        return 1

    _check_stage_a_isolation(ci, quality, errors)
    _check_quality_audit(ci, quality, errors)
    _check_ruff_scope(quality, errors)
    _check_condition_tensor(ci, errors)

    print("Allowed certification cell statuses:", ", ".join(sorted(ALLOWED_CELL_STATUSES)))
    if errors:
        for e in errors:
            _fail(e)
        return 1
    print("OK: CI/Quality gate ownership boundaries are consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
