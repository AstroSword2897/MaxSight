"""CLI behaviour of scripts/data_sources.py and scripts/validate_data.py (subprocess, no network)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from ml.datasets.manifest import BoxRecord, ImageRecord, write_manifest
from PIL import Image

REPO = Path(__file__).resolve().parents[2]


def _run(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(REPO / "scripts" / script), *args],  # noqa: S603
                          capture_output=True, text=True, timeout=60, check=False)


def test_status_lists_all_eight_datasets_as_configured(tmp_path):
    out = _run("data_sources.py", "--data-root", str(tmp_path), "status")
    assert out.returncode == 0
    for sid in ("coco2017", "open_images_v7", "mapillary_vistas_v2", "bdd100k", "sanpo", "diode", "pie", "ego4d"):
        assert sid in out.stdout
    assert out.stdout.count("CONFIGURED") == 8


def test_plan_sanpo_requires_explicit_sessions(tmp_path):
    out = _run("data_sources.py", "--data-root", str(tmp_path), "plan", "sanpo")
    assert out.returncode == 2 and "ERROR" in out.stderr and "Traceback" not in out.stderr


def test_plan_sanpo_is_a_dry_run_that_creates_nothing(tmp_path):
    out = _run("data_sources.py", "--data-root", str(tmp_path), "plan", "sanpo", "--session", "abc")
    assert out.returncode == 0 and "DRY-RUN gcloud storage rsync" in out.stdout
    assert not (tmp_path / "raw").exists()


def test_fetch_diode_refuses_when_approval_is_below_the_plan_size(tmp_path):
    out = _run("data_sources.py", "--data-root", str(tmp_path), "fetch", "diode", "--approve-gb", "1")
    assert out.returncode == 2 and "Refusing" in out.stdout
    assert not (tmp_path / "raw").exists()


def test_fetch_sanpo_without_execute_only_prints(tmp_path):
    out = _run("data_sources.py", "--data-root", str(tmp_path), "fetch", "sanpo", "--session", "s", "--approve-gb", "5")
    assert out.returncode == 0 and "Dry run only" in out.stdout


def test_unknown_dataset_is_an_argparse_error(tmp_path):
    out = _run("data_sources.py", "--data-root", str(tmp_path), "plan", "imagenet")
    assert out.returncode == 2 and "unknown dataset" in out.stderr


def test_ego4d_video_without_filter_is_refused_cleanly(tmp_path):
    out = _run("data_sources.py", "--data-root", str(tmp_path), "plan", "ego4d", "--video")
    assert out.returncode == 2 and "refusing a full download" in out.stderr


def _dataset(tmp_path: Path, classes: list[str]) -> Path:
    import random

    rng = random.Random(0)
    recs = []
    (tmp_path / "images").mkdir()
    for i, cls in enumerate(classes):
        im = Image.new("RGB", (64, 48))
        im.putdata([(rng.randrange(256), rng.randrange(256), rng.randrange(256)) for _ in range(64 * 48)])
        im.save(tmp_path / "images" / f"{i}.png")
        recs.append(ImageRecord(f"t:{i}", "t", f"images/{i}.png", 64, 48, f"g:{i}",
                                (BoxRecord(cls, (4.0, 4.0, 40.0, 40.0)),)))
    manifest = tmp_path / "m.jsonl"
    write_manifest(recs, manifest)
    return manifest


def test_validate_data_passes_a_balanced_manifest_and_writes_a_report(tmp_path):
    manifest = _dataset(tmp_path, ["person", "chair", "door", "stairs", "bench", "couch", "table", "animal"] * 3)
    report = tmp_path / "report.json"
    out = _run("validate_data.py", "--manifest", str(manifest), "--root", str(tmp_path), "--split",
               "--report", str(report))
    assert out.returncode == 0 and "AUDIT PASSED" in out.stdout
    assert json.loads(report.read_text())["ok"] is True


def test_validate_data_fails_a_skewed_manifest(tmp_path):
    manifest = _dataset(tmp_path, ["person"] * 20)
    out = _run("validate_data.py", "--manifest", str(manifest), "--root", str(tmp_path), "--split")
    assert out.returncode == 1 and "AUDIT FAILED" in out.stdout and "class_share_exceeds" in out.stdout


@pytest.mark.parametrize("script", ["train_detector.py", "evaluate_detector.py", "export_model.py",
                                    "generate_golden_vectors.py", "validate_release.py"])
def test_unimplemented_scripts_fail_loudly_never_pretend_success(script):
    out = _run(script)
    assert out.returncode != 0 and "not implemented" in out.stderr
