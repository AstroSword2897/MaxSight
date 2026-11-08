"""Ticket 309: an immutable, versioned split manifest built from whole sessions, with at least one whole
environment held out as test, and a leakage check that fails when any group or held-out environment leaks.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from ml.datasets.manifest import BoxRecord, ImageRecord, write_manifest
from ml.datasets.split_manifest import (
    SplitManifestError,
    apply_split_manifest,
    build_split_manifest,
    check_leakage,
    read_split_manifest,
    write_split_manifest,
)

REPO = Path(__file__).resolve().parents[2]


def frames(session: str, environment: str, count: int = 3) -> list[ImageRecord]:
    """``count`` distinct frames of one recording session in one environment (cleaned records carry sha256)."""
    return [
        ImageRecord(
            image_id=f"{session}:{i}", source="maxsight", path=f"{session}/{i}.jpg", width=10, height=10,
            group_id=session, boxes=(BoxRecord("stairs", (0.0, 0.0, 5.0, 5.0)),),
            sha256=hashlib.sha256(f"{session}:{i}".encode()).hexdigest(), extra={"environment": environment},
        )
        for i in range(count)
    ]


def dataset() -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for n in range(30):
        records += frames(f"s{n}", environment=f"env{n % 5}")
    return records


def build(records=None, **overrides):
    kwargs = {"version": "1", "holdout_environments": ("env4",), "reason": "initial split"}
    kwargs.update(overrides)
    return build_split_manifest(dataset() if records is None else records, **kwargs)


# --- building ----------------------------------------------------------------------------------

def test_whole_sessions_go_to_exactly_one_split():
    split_records = apply_split_manifest(dataset(), build())
    assert check_leakage(split_records) == []
    by_session: dict[str, set[str]] = {}
    for record in split_records:
        by_session.setdefault(record.group_id, set()).add(record.split)
    assert all(len(splits) == 1 for splits in by_session.values())


def test_the_held_out_environment_is_entirely_test_holdout():
    split_records = apply_split_manifest(dataset(), build())
    held = {r.split for r in split_records if r.extra["environment"] == "env4"}
    others = {r.split for r in split_records if r.extra["environment"] != "env4"}
    assert held == {"test_holdout"}
    assert "test_holdout" not in others
    assert {"train", "val", "test"} <= others


def test_at_least_one_environment_must_be_held_out():
    with pytest.raises(SplitManifestError, match="hold out at least one environment"):
        build(holdout_environments=())


def test_a_held_out_environment_must_exist():
    with pytest.raises(SplitManifestError, match="not in the data"):
        build(holdout_environments=("nowhere",))


def test_every_record_needs_an_environment():
    records = dataset() + [frames("s99", "envX", 1)[0].with_(extra={})]
    with pytest.raises(SplitManifestError, match="no environment"):
        build(records)


def test_a_session_cannot_span_two_environments():
    records = dataset() + frames("s0", "env3", 1)  # s0 is already in env0
    with pytest.raises(SplitManifestError, match="more than one environment"):
        build(records)


def test_version_and_reason_are_required():
    with pytest.raises(SplitManifestError, match="version"):
        build(version="")
    with pytest.raises(SplitManifestError, match="reason"):
        build(reason="")


def test_the_same_inputs_give_the_same_content_hash():
    assert build().content_hash == build().content_hash
    assert build().content_hash != build(seed=1).content_hash


# --- immutability ------------------------------------------------------------------------------

def test_a_written_manifest_cannot_be_overwritten(tmp_path):
    path = tmp_path / "split_v1.json"
    write_split_manifest(build(), path)
    with pytest.raises(FileExistsError):
        write_split_manifest(build(reason="second attempt"), path)


def test_a_manifest_round_trips(tmp_path):
    path = tmp_path / "split_v1.json"
    manifest = build()
    write_split_manifest(manifest, path)
    assert read_split_manifest(path) == manifest


def test_an_edited_manifest_is_detected(tmp_path):
    path = tmp_path / "split_v1.json"
    write_split_manifest(build(), path)
    doc = json.loads(path.read_text())
    first_group = sorted(doc["assignments"])[0]
    doc["assignments"][first_group] = "train" if doc["assignments"][first_group] != "train" else "test"
    path.write_text(json.dumps(doc))
    with pytest.raises(SplitManifestError, match="content hash"):
        read_split_manifest(path)


# --- leakage check -----------------------------------------------------------------------------

def test_leakage_check_finds_a_session_in_two_splits():
    records = [r.with_(split="train") for r in frames("s1", "env0")]
    records[0] = records[0].with_(split="test")
    assert check_leakage(records) == ["group_in_multiple_splits:s1:['test', 'train']"]


def test_leakage_check_finds_a_held_out_environment_outside_test_holdout():
    records = [r.with_(split="test_holdout") for r in frames("s1", "env4")]
    records += [r.with_(split="train") for r in frames("s2", "env4")]
    assert check_leakage(records) == ["environment_split_across_holdout:env4:['test_holdout', 'train']"]


def test_applying_a_manifest_to_an_unknown_session_fails():
    with pytest.raises(SplitManifestError, match="not in the split manifest"):
        apply_split_manifest(frames("new-session", "env0"), build())


# --- command line ------------------------------------------------------------------------------

def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(REPO / "scripts" / "split_manifest.py"), *args],  # noqa: S603
                          capture_output=True, text=True, timeout=60, check=False)


def test_cli_creates_a_manifest_and_the_leakage_check_passes(tmp_path):
    manifest = tmp_path / "records.jsonl"
    write_manifest(dataset(), manifest)
    out_path = tmp_path / "split_v1.json"

    created = run_cli("create", "--manifest", str(manifest), "--version", "1", "--holdout-environment", "env4",
                      "--reason", "initial split", "--out", str(out_path))
    checked = run_cli("check", "--manifest", str(manifest), "--split-manifest", str(out_path))

    assert created.returncode == 0, created.stdout + created.stderr
    assert checked.returncode == 0 and "LEAKAGE CHECK PASSED" in checked.stdout


def test_cli_refuses_to_overwrite_an_existing_version(tmp_path):
    manifest = tmp_path / "records.jsonl"
    write_manifest(dataset(), manifest)
    out_path = tmp_path / "split_v1.json"
    args = ("create", "--manifest", str(manifest), "--version", "1", "--holdout-environment", "env4",
            "--reason", "initial split", "--out", str(out_path))
    run_cli(*args)

    again = run_cli(*args)

    assert again.returncode == 2 and "already exists" in again.stdout
