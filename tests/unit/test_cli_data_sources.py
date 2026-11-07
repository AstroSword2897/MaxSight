"""CLI behaviour of scripts/data_sources.py (subprocess, no network)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

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
