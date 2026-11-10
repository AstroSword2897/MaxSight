"""Ticket 312: safety gates accept only human-reviewed, non-synthetic ground truth.

SG-01 and SG-02 used to be unblocked by a bare --hazard-gt flag. Now ground truth must be supplied as a
manifest, and a single synthetic or non-reviewed record fails the gate run.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from ml.datasets.manifest import BoxRecord, ImageRecord, write_manifest
from ml.evaluation.ground_truth_guard import GroundTruthRejected, assert_human_reviewed

REPO = Path(__file__).resolve().parents[2]
GATE_SCRIPT = REPO / "scripts" / "infra" / "run_safety_gate_ci.py"


def reviewed_record(image_id: str = "maxsight:1", **changes) -> ImageRecord:
    record = ImageRecord(
        image_id=image_id,
        source="maxsight",
        path=f"frames/{image_id}.jpg",
        width=100,
        height=100,
        group_id="session:1",
        boxes=(BoxRecord("stairs", (0.0, 0.0, 10.0, 10.0)),),
        sha256="a" * 64,
        dataset_version="hazard-v1",
        licence_id="maxsight-internal",
        label_map_version="1.0.0",
        annotation_source="human_reviewed",
    )
    return record.with_(**changes)


def run_gate(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(GATE_SCRIPT), *args],  # noqa: S603
                          capture_output=True, text=True, timeout=60, check=False)


# --- the guard ---------------------------------------------------------------------------------

def test_human_reviewed_records_are_accepted():
    assert assert_human_reviewed([reviewed_record("a"), reviewed_record("b")]) == 2


@pytest.mark.parametrize("changes", [
    {"is_synthetic": True, "annotation_source": "synthetic"},
    {"annotation_source": "synthetic"},
    {"annotation_source": "heuristic"},
    {"annotation_source": "model_prediction"},
    {"annotation_source": "human_unreviewed"},
    {"annotation_source": "source_dataset"},
    {"annotation_source": ""},
])
def test_any_record_that_is_not_human_reviewed_is_rejected(changes):
    records = [reviewed_record("good"), reviewed_record("bad", **changes)]
    with pytest.raises(GroundTruthRejected, match="bad"):
        assert_human_reviewed(records)


def test_a_synthetic_flag_is_rejected_even_if_marked_reviewed():
    with pytest.raises(GroundTruthRejected, match="synthetic"):
        assert_human_reviewed([reviewed_record(is_synthetic=True)])


def test_no_records_is_not_ground_truth():
    with pytest.raises(GroundTruthRejected, match="no records"):
        assert_human_reviewed([])


# --- the gate runner ---------------------------------------------------------------------------

def test_gate_run_fails_on_a_synthetic_record(tmp_path):
    manifest = tmp_path / "gt.jsonl"
    write_manifest([reviewed_record("ok"), reviewed_record("fake", is_synthetic=True,
                                                            annotation_source="synthetic")], manifest)

    out = run_gate("--output", str(tmp_path / "m.json"), "--ground-truth-manifest", str(manifest))

    assert out.returncode == 2
    assert "GROUND TRUTH REJECTED" in out.stdout and "fake" in out.stdout
    assert not (tmp_path / "m.json").exists()


def test_hazard_gt_flag_alone_no_longer_unblocks_the_gates(tmp_path):
    out = run_gate("--output", str(tmp_path / "m.json"), "--hazard-gt")

    assert out.returncode == 2
    assert "--ground-truth-manifest" in out.stderr


def test_without_ground_truth_sg01_and_sg02_stay_blocked(tmp_path):
    out = run_gate("--output", str(tmp_path / "m.json"))

    assert out.returncode == 1
    assert "blocked=" in out.stdout and "blocked=0" not in out.stdout
