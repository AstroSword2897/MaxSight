"""CLI behaviour of scripts/validate_data.py (subprocess, no network)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

from ml.datasets.manifest import BoxRecord, ImageRecord, write_manifest

REPO = Path(__file__).resolve().parents[2]


def _run(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(REPO / "scripts" / script), *args],  # noqa: S603
                          capture_output=True, text=True, timeout=60, check=False)



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


def test_validate_data_writes_the_validated_marker_only_when_the_audit_passes(tmp_path):
    manifest = _dataset(tmp_path, ["person", "chair", "door", "stairs", "bench", "couch", "table", "animal"] * 3)
    data_root = tmp_path / "datasets"
    out = _run("validate_data.py", "--manifest", str(manifest), "--root", str(tmp_path), "--split",
               "--dataset-id", "open_images_v7", "--data-root", str(data_root))
    marker = data_root / "processed" / "open_images_v7" / "VALIDATED.json"
    assert out.returncode == 0 and json.loads(marker.read_text())["ok"] is True


def test_failed_audit_removes_a_stale_validated_marker(tmp_path):
    manifest = _dataset(tmp_path, ["person"] * 20)
    marker = tmp_path / "datasets" / "processed" / "open_images_v7" / "VALIDATED.json"
    marker.parent.mkdir(parents=True)
    marker.write_text('{"ok": true}')
    out = _run("validate_data.py", "--manifest", str(manifest), "--root", str(tmp_path), "--split",
               "--dataset-id", "open_images_v7", "--data-root", str(tmp_path / "datasets"))
    assert out.returncode == 1 and not marker.exists()
