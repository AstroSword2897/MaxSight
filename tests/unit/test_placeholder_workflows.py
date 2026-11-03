"""Placeholder workflows must fail when dispatched: a green run would read as a passing gate that never ran."""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"
PLACEHOLDERS = ["model-evaluation", "release-gates"]


@pytest.mark.parametrize("name", PLACEHOLDERS)
def test_placeholder_workflow_is_manual_only(name):
    doc = yaml.safe_load((WORKFLOWS / f"{name}.yml").read_text())
    triggers = doc.get("on", doc.get(True))  # PyYAML reads the bare key `on` as boolean True
    assert set(triggers) == {"workflow_dispatch"}


@pytest.mark.parametrize("name", PLACEHOLDERS)
def test_placeholder_workflow_fails_until_the_gate_exists(name):
    doc = yaml.safe_load((WORKFLOWS / f"{name}.yml").read_text())
    runs = [s["run"] for job in doc["jobs"].values() for s in job["steps"] if "run" in s]
    assert runs and all("exit 1" in r for r in runs), "a placeholder gate must not exit 0"
