"""The same image (byte-identical or a near copy) must never sit in two splits.

Group and environment checks cannot see two sessions that contain the same picture. The split
manifest refuses to freeze such data, check_leakage reports it, and the pre-training audit
compares exact hashes as well as perceptual ones.
"""

from __future__ import annotations

import hashlib

import pytest
from ml.datasets.audit import audit
from ml.datasets.label_map import LabelMap
from ml.datasets.manifest import BoxRecord, ImageRecord
from ml.datasets.split_manifest import SplitManifestError, build_split_manifest, check_leakage


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def rec(image_id: str, group: str, split: str = "", env: str = "env0", **kw) -> ImageRecord:
    kw.setdefault("sha256", _sha(image_id))
    return ImageRecord(
        image_id=image_id, source="maxsight", path=f"{image_id}.jpg", width=10, height=10, group_id=group,
        boxes=(BoxRecord("stairs", (0.0, 0.0, 5.0, 5.0)),), split=split, extra={"environment": env}, **kw,
    )


# --- check_leakage --------------------------------------------------------------------------------

def test_identical_bytes_in_two_splits_is_leakage():
    records = [rec("a", "s1", "train"), rec("b", "s2", "test", sha256=_sha("a"))]
    assert check_leakage(records) == ["same_content_across_splits:a:b"]


def test_near_copy_in_two_splits_is_leakage():
    records = [rec("a", "s1", "train", dhash=f"{0xF0F0:016x}"), rec("b", "s2", "test", dhash=f"{0xF0F1:016x}")]
    assert check_leakage(records) == ["near_duplicate_across_splits:a:b"]


def test_identical_bytes_in_one_split_is_not_leakage():
    records = [rec("a", "s1", "train"), rec("b", "s2", "train", sha256=_sha("a"))]
    assert check_leakage(records) == []


def test_a_record_without_any_content_hash_cannot_be_cleared():
    records = [rec("a", "s1", "train"), rec("b", "s2", "test", sha256="")]
    assert check_leakage(records) == ["missing_content_hash:b"]


# --- build_split_manifest -------------------------------------------------------------------------

def _sessions(shared_sha: str = "", shared_dhash: str = "") -> list[ImageRecord]:
    records = [rec(f"s{n}:{i}", f"s{n}", env=f"env{n % 5}") for n in range(30) for i in range(3)]
    if shared_sha:
        records.append(rec("s7:dup", "s7", env="env2", sha256=shared_sha))
    if shared_dhash:
        records[0] = records[0].with_(dhash=f"{0xABCD:016x}")
        records.append(rec("s7:near", "s7", env="env2", dhash=shared_dhash))
    return records


@pytest.mark.parametrize("seed", range(5))
def test_build_refuses_groups_that_share_an_image_whatever_the_seed(seed):
    records = _sessions(shared_sha=_sha("s0:0"))
    with pytest.raises(SplitManifestError, match="content_shared_across_groups:s0:0:s7:dup"):
        build_split_manifest(records, version="1", holdout_environments=("env4",), reason="r", seed=seed)


def test_build_refuses_groups_that_share_a_near_copy():
    records = _sessions(shared_dhash=f"{0xABCC:016x}")
    with pytest.raises(SplitManifestError, match="content_shared_across_groups"):
        build_split_manifest(records, version="1", holdout_environments=("env4",), reason="r")


def test_build_refuses_records_without_a_content_hash():
    records = _sessions()
    records[5] = records[5].with_(sha256="")
    with pytest.raises(SplitManifestError, match="missing_content_hash"):
        build_split_manifest(records, version="1", holdout_environments=("env4",), reason="r")


def test_build_accepts_duplicates_inside_one_group():
    records = _sessions() + [rec("s3:again", "s3", env="env3", sha256=_sha("s3:0"))]
    build_split_manifest(records, version="1", holdout_environments=("env4",), reason="r")


# --- pre-training audit ---------------------------------------------------------------------------

def test_audit_compares_exact_hashes_when_no_perceptual_hash_is_present():
    records = [rec("a", "s1", "train"), rec("b", "s2", "test", sha256=_sha("a"))]
    res = audit(records, LabelMap.load(), max_class_share=1.0)
    assert "same_content_across_splits:a:b" in res.errors


def test_audit_warns_when_split_records_carry_no_content_hash():
    records = [rec("a", "s1", "train", sha256=""), rec("b", "s2", "test")]
    res = audit(records, LabelMap.load(), max_class_share=1.0)
    assert "content_unverified:1_split_records_without_sha256_or_dhash" in res.warnings
