"""End-to-end tests for ml.datasets.acquisition.fetch against a local HTTP server (no internet)."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import threading
import zipfile
from functools import partial
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from ml.datasets.acquisition import fetch, plan
from ml.datasets.sources import ACQUIRED_MARKER, load_registry


class _Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, files: dict[str, bytes], log: list, faults: dict[str, int], **kwargs):
        self._files, self._log, self._faults = files, log, faults
        super().__init__(*args, **kwargs)

    def log_message(self, *args):  # silence
        pass

    def do_GET(self):  # noqa: N802
        body = self._files.get(self.path.lstrip("/"))
        if body is None:
            self.send_error(404)
            return
        rng = self.headers.get("Range")
        self._log.append((self.path, rng))
        if rng:
            start = int(rng.split("=")[1].split("-")[0])
            if start >= len(body):
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{len(body)}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(body) - 1}/{len(body)}")
            body = body[start:]
        else:
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self._faults.get(self.path.lstrip("/"), 0) > 0:  # promise the full body, send half, hang up
            self._faults[self.path.lstrip("/")] -= 1
            self.wfile.write(body[: len(body) // 2])
            self.close_connection = True
            return
        self.wfile.write(body)


@pytest.fixture
def server():
    files: dict[str, bytes] = {}
    log: list = []
    faults: dict[str, int] = {}
    httpd = HTTPServer(("127.0.0.1", 0), partial(_Handler, files=files, log=log, faults=faults))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}", files, log, faults
    httpd.shutdown()
    httpd.server_close()


def _tar_gz(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _zip(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _spec(base, val: bytes, md5: str | None):
    return {"archives": {
        "val": {"url": f"{base}/val.tar.gz", "md5": md5},
        "data_list": {"url": f"{base}/data_list.zip", "md5": None},
    }}


def test_diode_fetch_downloads_verifies_extracts_and_marks_verified(server, tmp_path):
    base, files, _, _ = server
    val = _tar_gz({"val/indoors/scene_1/scan_1/a.png": b"png", "val/indoors/scene_1/scan_1/a_depth.npy": b"d"})
    files["val.tar.gz"] = val
    files["data_list.zip"] = _zip({"val.csv": b"a\n"})
    info = fetch.fetch_diode(_spec(base, val, hashlib.md5(val).hexdigest()), tmp_path, include=("val",))
    assert info["verified"] is True
    assert (tmp_path / "extracted/val/indoors/scene_1/scan_1/a.png").read_bytes() == b"png"
    marker = json.loads((tmp_path / ACQUIRED_MARKER).read_text())
    assert marker["files"]["val.tar.gz"]["md5_verified"] is True


def test_diode_fetch_without_published_checksum_is_downloaded_but_not_verified(server, tmp_path):
    base, files, _, _ = server
    val = _tar_gz({"val/x.png": b"1"})
    files["val.tar.gz"] = val
    files["data_list.zip"] = _zip({"val.csv": b"a\n"})
    info = fetch.fetch_diode(_spec(base, val, hashlib.md5(val).hexdigest()), tmp_path)
    assert info["verified"] is False  # data_list.zip has no published MD5


def test_diode_fetch_refuses_md5_mismatch_and_writes_no_marker(server, tmp_path):
    base, files, _, _ = server
    files["val.tar.gz"] = _tar_gz({"val/x.png": b"1"})
    with pytest.raises(RuntimeError, match="MD5 mismatch"):
        fetch.fetch_diode(_spec(base, b"", "0" * 32), tmp_path, include=("val",))
    assert not (tmp_path / ACQUIRED_MARKER).exists()
    # the bad file is moved aside (kept for inspection) so the next run downloads again instead of reusing it
    assert not (tmp_path / "archives" / "val.tar.gz").exists()
    assert (tmp_path / "archives" / "val.tar.gz.md5-mismatch").exists()


def test_diode_fetch_recovers_after_a_corrupt_download(server, tmp_path):
    base, files, _, _ = server
    good = _tar_gz({"val/x.png": b"1"})
    files["val.tar.gz"] = _tar_gz({"val/x.png": b"corrupt"})
    with pytest.raises(RuntimeError, match="MD5 mismatch"):
        fetch.fetch_diode(_spec(base, good, hashlib.md5(good).hexdigest()), tmp_path, include=("val",))
    files["val.tar.gz"] = good  # server fixed
    info = fetch.fetch_diode(_spec(base, good, hashlib.md5(good).hexdigest()), tmp_path, include=("val",))
    assert info["verified"] is True


def test_download_resumes_a_partial_file_with_a_range_request(server, tmp_path):
    base, files, log, _ = server
    body = bytes(range(256)) * 400
    files["big.bin"] = body
    dest = tmp_path / "big.bin"
    (tmp_path / "big.bin.part").write_bytes(body[:1000])
    fetch.download_resumable(f"{base}/big.bin", dest)
    assert dest.read_bytes() == body
    assert log[0][1] == "bytes=1000-"


def test_download_never_replaces_a_finished_file(server, tmp_path):
    base, files, log, _ = server
    files["a.bin"] = b"new"
    dest = tmp_path / "a.bin"
    dest.write_bytes(b"original")
    fetch.download_resumable(f"{base}/a.bin", dest)
    assert dest.read_bytes() == b"original" and log == []


def test_download_raises_after_bounded_retries(server, tmp_path, monkeypatch):
    base, _, _, _ = server
    monkeypatch.setattr(fetch.time, "sleep", lambda _s: None)
    with pytest.raises(RuntimeError, match="after 2 attempts"):
        fetch.download_resumable(f"{base}/missing.bin", tmp_path / "m.bin", retries=2)


def test_safe_extract_zip_rejects_path_traversal(tmp_path):
    evil = tmp_path / "evil.zip"
    evil.write_bytes(_zip({"../escape.txt": b"x"}))
    with pytest.raises(ValueError, match="unsafe"):
        fetch.safe_extract_zip(evil, tmp_path / "out")


def test_run_plan_dry_run_never_spawns_a_process(monkeypatch, capsys):
    monkeypatch.setattr(fetch.subprocess, "run", lambda *a, **k: pytest.fail("spawned"))
    pl = plan.sanpo_plan(load_registry()["sanpo"], ["s1"])
    assert fetch.run_plan(pl, dry_run=True) == 0
    assert "DRY-RUN gcloud storage rsync" in capsys.readouterr().out


def test_run_plan_stops_at_first_failing_step(monkeypatch):
    calls = []

    class R:
        def __init__(self, rc): self.returncode = rc

    def fake(argv, check):
        calls.append(argv)
        return R(3 if len(calls) == 1 else 0)

    monkeypatch.setattr(fetch.subprocess, "run", fake)
    pl = plan.sanpo_plan(load_registry()["sanpo"], ["s1", "s2"])
    assert fetch.run_plan(pl, dry_run=False) == 3
    assert len(calls) == 1


def test_truncated_download_is_never_renamed_to_the_final_file(server, tmp_path, monkeypatch):
    base, files, _, faults = server
    monkeypatch.setattr(fetch.time, "sleep", lambda _s: None)
    files["big.bin"] = bytes(range(256)) * 400
    faults["big.bin"] = 99  # every response is cut short
    with pytest.raises(RuntimeError, match="after 2 attempts"):
        fetch.download_resumable(f"{base}/big.bin", tmp_path / "big.bin", retries=2)
    assert not (tmp_path / "big.bin").exists()


def test_truncated_download_resumes_from_the_partial_file(server, tmp_path, monkeypatch):
    base, files, log, faults = server
    monkeypatch.setattr(fetch.time, "sleep", lambda _s: None)
    body = bytes(range(256)) * 400
    files["big.bin"] = body
    faults["big.bin"] = 1  # first response is cut short, the retry is complete
    fetch.download_resumable(f"{base}/big.bin", tmp_path / "big.bin")
    assert (tmp_path / "big.bin").read_bytes() == body
    assert log[1][1] and log[1][1].startswith("bytes=")  # the retry resumed with a Range request


def test_complete_part_file_is_promoted_instead_of_failing_on_416(server, tmp_path, monkeypatch):
    """Killed after the last chunk but before the rename: the Range request gets 416 on every retry."""
    base, files, _, _ = server
    monkeypatch.setattr(fetch.time, "sleep", lambda _s: None)
    body = b"x" * 5000
    files["done.bin"] = body
    (tmp_path / "done.bin.part").write_bytes(body)
    fetch.download_resumable(f"{base}/done.bin", tmp_path / "done.bin", retries=2)
    assert (tmp_path / "done.bin").read_bytes() == body and not (tmp_path / "done.bin.part").exists()


def test_part_file_longer_than_the_remote_restarts_the_download(server, tmp_path, monkeypatch):
    base, files, _, _ = server
    monkeypatch.setattr(fetch.time, "sleep", lambda _s: None)
    files["shrunk.bin"] = b"new"
    (tmp_path / "shrunk.bin.part").write_bytes(b"old content that is longer")
    fetch.download_resumable(f"{base}/shrunk.bin", tmp_path / "shrunk.bin", retries=2)
    assert (tmp_path / "shrunk.bin").read_bytes() == b"new"


def test_coco_annotations_zip_is_kept_on_disk_and_downloaded_once(server, tmp_path):
    from ml.datasets.acquisition import coco

    base, files, log, _ = server
    files["ann.zip"] = _zip({"annotations/instances_val2017.json": b"{}", "annotations/instances_train2017.json": b"[]"})
    dest = tmp_path / "annotations"
    assert coco.fetch_annotations(dest, "val2017", url=f"{base}/ann.zip").read_bytes() == b"{}"
    assert coco.fetch_annotations(dest, "train2017", url=f"{base}/ann.zip").read_bytes() == b"[]"
    assert len(log) == 1
