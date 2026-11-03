"""Execution for sources that can be fetched automatically: resumable HTTP + checksum, and plan subprocess runs."""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

from ml.datasets.acquisition.plan import Plan
from ml.datasets.sources import ACQUIRED_MARKER

log = logging.getLogger(__name__)
_UA = {"User-Agent": "maxsight-data/1.0"}
_CHUNK = 1 << 20


def file_hash(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def download_resumable(url: str, dest: Path, retries: int = 4) -> Path:
    """Download ``url`` to ``dest`` using a ``.part`` file and HTTP Range resume. Never replaces a finished file."""
    if dest.exists():
        log.info("exists, not replacing: %s", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={**_UA, **({"Range": f"bytes={have}-"} if have else {})})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 - registry-pinned URL
                if have and resp.status != 206:
                    have = 0  # server ignored Range: restart
                with part.open("ab" if have else "wb") as out:
                    while chunk := resp.read(_CHUNK):
                        out.write(chunk)
            part.replace(dest)
            return dest
        except Exception as exc:  # noqa: BLE001 - retried then raised
            log.warning("download attempt %d failed: %s", attempt + 1, exc)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"download failed after {retries} attempts: {url}")


def verify_md5(path: Path, expected: str | None) -> bool:
    return expected is None or file_hash(path, "md5") == expected


def safe_extract_tar(archive: Path, dest: Path) -> None:
    root = dest.resolve()
    with tarfile.open(archive) as tf:
        for m in tf.getmembers():
            target = (dest / m.name).resolve()
            if not target.is_relative_to(root) or m.issym() or m.islnk():
                raise ValueError(f"unsafe archive member: {m.name}")
        tf.extractall(dest)  # noqa: S202 - members validated above


def safe_extract_zip(archive: Path, dest: Path) -> None:
    root = dest.resolve()
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            if not (dest / name).resolve().is_relative_to(root):
                raise ValueError(f"unsafe archive member: {name}")
        zf.extractall(dest)


def fetch_diode(src_raw: dict, raw: Path, include: tuple[str, ...] = ("val", "data_list")) -> dict:
    """Fetch DIODE validation archives, verify MD5 where published, extract, write the ACQUIRED marker."""
    files: dict[str, dict] = {}
    all_verified = True
    for key in include:
        spec = src_raw["archives"][key]
        dest = download_resumable(spec["url"], raw / "archives" / Path(spec["url"]).name)
        ok = verify_md5(dest, spec.get("md5"))
        if not ok:
            raise RuntimeError(f"MD5 mismatch for {dest.name}; leaving file in place for inspection")
        all_verified &= spec.get("md5") is not None
        files[dest.name] = {"size": dest.stat().st_size, "sha256": file_hash(dest, "sha256"),
                            "md5_verified": spec.get("md5") is not None}
        extractor = safe_extract_tar if dest.name.endswith(".tar.gz") else safe_extract_zip
        extractor(dest, raw / "extracted")
    info = {"verified": all_verified, "files": files, "acquired_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (raw / ACQUIRED_MARKER).write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def run_plan(plan: Plan, dry_run: bool = True) -> int:
    """Run each argv step. With ``dry_run`` only print. Fails on the first non-zero exit."""
    for step in plan.argv:
        print(("DRY-RUN " if dry_run else "RUN ") + " ".join(step))
        if not dry_run:
            rc = subprocess.run(step, check=False).returncode  # noqa: S603 - argv list, no shell
            if rc:
                return rc
    return 0
