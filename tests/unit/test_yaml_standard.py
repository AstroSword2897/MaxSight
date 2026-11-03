"""House style for configuration YAML (plan step B).

Rules: a leading ``#`` header comment, a top-level ``schema_version: "MAJOR.MINOR.PATCH"`` string,
snake_case top-level keys, and no flow-style mappings. Files that do not comply yet are listed in
``KNOWN_GAPS`` as strict xfails: fixing a file without removing its entry fails the suite.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO = Path(__file__).resolve().parents[2]
SCOPES = ("configs", "ml/config", "ml/training/configs")
EXTRA = ("scripts/product/deprecations.yaml",)
SEMVER = re.compile(r"^\d+\.\d+\.\d+$")
SNAKE = re.compile(r"^[a-z][a-z0-9_]*$")
FORBIDDEN_SENSORS = ("imu", "lidar", "gps", "accelerometer", "gyroscope")

KNOWN_GAPS: dict[str, str] = {}  # file -> reason; add an entry only for a deliberate, tracked gap


def _files() -> list[str]:
    found = {p.relative_to(REPO).as_posix() for s in SCOPES for p in (REPO / s).rglob("*.yaml")}
    return sorted(found | set(EXTRA))


def _check(rel: str) -> None:
    text = (REPO / rel).read_text(encoding="utf-8")
    doc = yaml.safe_load(text)
    assert isinstance(doc, dict), "top level must be a mapping"
    first = next(line for line in text.splitlines() if line.strip())
    assert first.startswith("#"), "missing header comment"
    version = doc.get("schema_version")
    assert isinstance(version, str) and SEMVER.match(version), f"schema_version={version!r}"
    assert not re.search(r":\s*\{", text), "flow-style mapping"
    bad = [k for k in doc if not SNAKE.match(str(k))]
    assert not bad, f"non-snake_case keys {bad}"


def _param(rel: str):
    marks = [pytest.mark.xfail(strict=True, reason=KNOWN_GAPS[rel])] if rel in KNOWN_GAPS else []
    return pytest.param(rel, marks=marks, id=rel)


@pytest.mark.parametrize("rel", [_param(r) for r in _files()])
def test_yaml_follows_house_style(rel):
    _check(rel)


def test_known_gaps_reference_real_files():
    assert set(KNOWN_GAPS) <= set(_files())


@pytest.mark.parametrize("rel", _files())
def test_yaml_parses_and_is_not_empty(rel):
    assert yaml.safe_load((REPO / rel).read_text(encoding="utf-8"))


def _keys(node):
    if isinstance(node, dict):
        for k, v in node.items():
            yield str(k)
            yield from _keys(v)
    elif isinstance(node, list):
        for item in node:
            yield from _keys(item)


@pytest.mark.parametrize("rel", [r for r in _files() if "therapy" not in r])
def test_no_config_key_names_a_sensor_outside_the_camera_only_product(rel):
    doc = yaml.safe_load((REPO / rel).read_text(encoding="utf-8"))
    hits = [k for k in _keys(doc) if any(re.search(rf"\b{w}\b", k.lower()) for w in FORBIDDEN_SENSORS)]
    assert not hits, hits


def test_target_workflow_files_exist():
    for name in ("ci", "model-evaluation", "release-gates"):
        assert (REPO / ".github" / "workflows" / f"{name}.yml").is_file(), name
