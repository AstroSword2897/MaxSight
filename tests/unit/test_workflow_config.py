"""Lint the GitHub Actions workflows so CI wiring mistakes fail fast (torch-free)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOWS = sorted((REPO / ".github" / "workflows").glob("*.yml"))
SHA = re.compile(r"^[0-9a-f]{40}$")


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_workflow_parses_and_has_jobs(path):
    data = _load(path)
    assert isinstance(data.get("jobs"), dict) and data["jobs"], path.name


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_needs_reference_existing_jobs(path):
    jobs = _load(path)["jobs"]
    for job_id, job in jobs.items():
        needs = job.get("needs", [])
        for dep in [needs] if isinstance(needs, str) else needs:
            assert dep in jobs, f"{path.name}: {job_id} needs unknown job {dep!r}"


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_third_party_actions_are_sha_pinned(path):
    for job_id, job in _load(path)["jobs"].items():
        refs = [job.get("uses", "")] + [s.get("uses", "") for s in job.get("steps", [])]
        for ref in filter(None, refs):
            if ref.startswith("./"):
                continue
            name, _, version = ref.partition("@")
            assert SHA.match(version), f"{path.name}:{job_id}: {name} is not pinned to a commit SHA ({version!r})"


def test_ci_summary_aggregates_every_other_job():
    jobs = _load(REPO / ".github/workflows/ci.yml")["jobs"]
    summary = jobs["ci-summary"]
    missing = set(jobs) - {"ci-summary"} - set(summary["needs"])
    assert not missing, f"ci-summary does not wait for: {sorted(missing)}"
    results = summary["steps"][0]["env"]["RESULTS"]
    unreported = [j for j in jobs if j not in ("ci-summary", "detect-changes") and f"{j}=" not in results]
    assert not unreported, f"ci-summary RESULTS omits: {unreported}"


def test_unit_tests_cpu_deselects_benchmarks():
    cmds = _load(REPO / ".github/workflows/ci.yml")["jobs"]["unit-tests-cpu"]["with"]["commands"]
    assert 'not benchmark' in cmds


def test_regression_guard_job_is_wired_and_torch_free():
    job = _load(REPO / ".github/workflows/ci.yml")["jobs"]["regression-guards"]
    assert job["with"]["profile"] == "contracts"
    assert "tests/unit" in job["with"]["commands"]


def test_pre_sagemaker_gate_job_asserts_torch_is_absent():
    steps = _load(REPO / ".github/workflows/quality.yml")["jobs"]["pre-sagemaker-gate"]["steps"]
    runs = " ".join(s.get("run", "") for s in steps)
    assert "find_spec('torch')" in runs and "pre_sagemaker_gate.py" in runs


def _script_invocations(script: str) -> list[list[str]]:
    """Every `python <script> ...` call in ci.yml, as argv after the script path."""
    import shlex

    calls = []
    for job_id, job in _load(REPO / ".github/workflows/ci.yml")["jobs"].items():
        commands = (job.get("with") or {}).get("commands", "")
        for line in commands.replace("\\\n", " ").splitlines():
            tokens = shlex.split(line)
            if script in tokens:
                calls.append((job_id, tokens[tokens.index(script) + 1 :]))
    return calls


def test_ci_sagemaker_deploy_calls_parse(monkeypatch):
    """Top-level options (--bucket, --role) must precede the subcommand, or argparse exits 2."""
    import importlib.util
    import sys

    script = "scripts/ops/sagemaker_deploy.py"
    calls = _script_invocations(script)
    assert calls, f"ci.yml no longer invokes {script}; update this test"
    spec = importlib.util.spec_from_file_location("sagemaker_deploy_cli", REPO / script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for job_id, argv in calls:
        monkeypatch.setattr(sys, "argv", [script, *argv])
        try:
            module.parse_args()
        except SystemExit as exc:
            pytest.fail(f"ci.yml:{job_id}: {script} rejects its arguments {argv} (exit {exc.code})")
