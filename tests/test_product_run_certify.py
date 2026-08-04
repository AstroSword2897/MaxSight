"""Product run.py certify tests (MAXS-301d)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_certify_refuses_sign_when_blocked(tmp_path: Path) -> None:
    out = tmp_path / "m.json"
    artifact = tmp_path / "a.bin"
    artifact.write_bytes(b"x")
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "product" / "run.py"),
            "certify",
            "--output",
            str(out),
            "--artifact",
            str(artifact),
            "--sign",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["all_passed"] is False
    assert not (tmp_path / "a.bin.sig").exists()
