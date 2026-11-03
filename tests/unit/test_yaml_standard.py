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
SEMVER = re.compile(r"^\d+\.\d+\.\d+$")
SNAKE = re.compile(r"^[a-z][a-z0-9_]*$")
FORBIDDEN_SENSORS = ("imu", "lidar", "gps", "accelerometer", "gyroscope")

KNOWN_GAPS: dict[str, str] = {}  # file -> reason; add an entry only for a deliberate, tracked gap


def _files() -> list[str]:
    roots = [REPO / s for s in SCOPES]
    found = {p for r in roots for pattern in ("*.yaml", "*.yml") for p in r.rglob(pattern)}
    found |= {p for pattern in ("*.yaml", "*.yml") for p in (REPO / "scripts" / "product").glob(pattern)}
    return sorted(p.relative_to(REPO).as_posix() for p in found)


def has_flow_mapping(text: str) -> bool:
    """True for any `{...}` mapping in YAML flow style, including list items; comments and strings never count."""
    return any(isinstance(ev, yaml.MappingStartEvent) and ev.flow_style for ev in yaml.parse(text))


@pytest.mark.parametrize(("text", "flagged"), [
    ("a: {b: 1}\n", True),
    ("items:\n  - {name: x, gb: 1}\n", True),
    ("# e.g. key: {..}\na: 1\n", False),
    ('a: "literal {braces}"\n', False),
    ("a: [x, y]\n", False),
])
def test_flow_mapping_detector(text, flagged):
    assert has_flow_mapping(text) is flagged


def _check(rel: str) -> None:
    text = (REPO / rel).read_text(encoding="utf-8")
    doc = yaml.safe_load(text)
    assert isinstance(doc, dict), "top level must be a mapping"
    first = next(line for line in text.splitlines() if line.strip())
    assert first.startswith("#"), "missing header comment"
    version = doc.get("schema_version")
    assert isinstance(version, str) and SEMVER.match(version), f"schema_version={version!r}"
    assert not has_flow_mapping(text), "flow-style mapping"
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


def names_a_sensor(key: str) -> bool:
    """True when a config key contains a forbidden sensor as a whole token (`imu_enabled`, `use-gps`, `Lidar`)."""
    return bool(set(re.split(r"[^a-z0-9]+", key.lower())) & set(FORBIDDEN_SENSORS))


@pytest.mark.parametrize("key", ["imu", "imu_enabled", "use_gps", "gps-lat", "Lidar", "enable.accelerometer.rate"])
def test_sensor_detector_catches_snake_kebab_and_dotted_keys(key):
    assert names_a_sensor(key)


@pytest.mark.parametrize("key", ["simulation", "imutable_flag", "gpsx", "rate_limits", "camera_id"])
def test_sensor_detector_ignores_lookalike_substrings(key):
    assert not names_a_sensor(key)


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
    hits = [k for k in _keys(doc) if names_a_sensor(k)]
    assert not hits, hits


def test_target_workflow_files_exist():
    for name in ("ci", "model-evaluation", "release-gates"):
        assert (REPO / ".github" / "workflows" / f"{name}.yml").is_file(), name
