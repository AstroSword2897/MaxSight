"""Raw-data integrity check: is every image a manifest points to present, readable and unchanged?

COCO and Open Images publish no per-image checksums, so this check cannot prove a file matches
the source. It proves something narrower, and says so in its report:

  * the file exists,
  * it decodes as an image,
  * its SHA-256 equals the hash recorded when it was downloaded.

A manifest with no records fails, because an empty list is not evidence of a download.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from ml.datasets.acquisition.fetch import file_hash
from ml.datasets.manifest import read_manifest

# Written into the ACQUIRED marker so nobody mistakes a local check for a source checksum.
VERIFICATION_METHOD = "local: file present, decodes, sha256 matches the hash recorded at download"


@dataclass
class IntegrityReport:
    checked: int = 0
    passed: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)  # (relative path, reason)

    @property
    def ok(self) -> bool:
        """True only if at least one file was checked and none failed."""
        return self.checked > 0 and not self.failures

    def as_dict(self) -> dict:
        return {
            "method": VERIFICATION_METHOD,
            "checked": self.checked,
            "passed": self.passed,
            "failed": len(self.failures),
            "first_failures": self.failures[:20],
        }


def check_file(path: Path, expected_sha256: str) -> str:
    """Return "" if the file is good, otherwise the reason it failed."""
    if not path.is_file():
        return "missing"

    try:
        with Image.open(path) as image:
            image.verify()  # reads the whole file and raises if it is truncated or not an image
    except Exception as exc:  # noqa: BLE001 - any decode error is a failure reason, not a crash
        return f"undecodable: {type(exc).__name__}"

    if not expected_sha256:
        return "no recorded sha256"
    if file_hash(path, "sha256") != expected_sha256:
        return "sha256 mismatch"
    return ""


def verify_raw(manifest: Path | str, root: Path | str) -> IntegrityReport:
    """Check every record in ``manifest``; image paths are relative to ``root``."""
    root = Path(root)
    report = IntegrityReport()

    for record in read_manifest(manifest):
        report.checked += 1
        reason = check_file(root / record.path, record.sha256)
        if reason:
            report.failures.append((record.path, reason))
        else:
            report.passed += 1

    return report
