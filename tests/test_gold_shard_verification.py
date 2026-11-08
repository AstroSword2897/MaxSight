"""Training must verify gold shards against the SHA-256 recorded in their meta.

The meta schema requires a sha256 per shard, but the training loader never asked for verification,
and relative shard URIs were resolved to absolute paths while the meta kept the relative ones, so
even an explicit verification found no hash and skipped every shard with a warning.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from ml.data import data_pipeline
from ml.data.data_pipeline import _gold_shard_uris_from_cfg
from ml.data.gold.builder import build_gold_manifest, write_manifest_meta
from ml.data.gold.dataNormalizationLayer import MaxSightListAdapter
from ml.data.gold.dataset import GoldManifestDataset
from ml.data.gold.io import GoldIOError
from ml.data.gold.label_mapper import LabelMapper
from ml.data.gold.schema import LABEL_SPACE_ACCESSIBILITY_622
from PIL import Image


def _artifact(repo: Path, name: str = "train") -> Path:
    """Build a one-image gold artifact whose meta lists its shard by a path relative to the meta file."""
    img_dir = repo / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color=(1, 2, 3)).save(img_dir / "a.jpg")
    ann = repo / f"{name}.json"
    ann.write_text(json.dumps([{"image_id": 0, "image_path": "a.jpg",
                                "objects": [{"category": "person", "box": [0.5, 0.5, 0.2, 0.2]}]}]))
    adapter = MaxSightListAdapter(ann, img_dir, repo, dataset_id="ds", version="v1", split=name)
    out = repo / "gold" / name
    # num_shards=1 writes exactly the file named by ``out``.
    summary = build_gold_manifest(adapter, mapper=LabelMapper(None, LABEL_SPACE_ACCESSIBILITY_622),
                                  out=out / "shard_00000.jsonl", repo_root=repo, source_annotation=str(ann),
                                  num_shards=1)
    meta_path = out / "meta.json"
    shards = [{**s, "uri": str(Path(s["uri"]).resolve().relative_to(out.resolve()))} for s in summary["shards"]]
    write_manifest_meta(meta_path, repo_root=repo, dataset_id="ds", version="v1", split=name,
                        label_space=LABEL_SPACE_ACCESSIBILITY_622, num_classes=622,
                        class_map_hash=summary["class_map_hash"], source_annotation=str(ann),
                        lines_written=summary["lines_written"], lines_skipped=summary["lines_skipped"],
                        shards=shards)
    return meta_path


def test_resolved_meta_names_the_same_shards_as_the_resolved_uris(tmp_path: Path):
    uris, meta = _gold_shard_uris_from_cfg(str(_artifact(tmp_path)), None, tmp_path, "data.gold_train")
    assert [s["uri"] for s in meta["shards"]] == uris


def test_a_tampered_shard_is_refused_through_pipeline_resolution(tmp_path: Path):
    uris, meta = _gold_shard_uris_from_cfg(str(_artifact(tmp_path)), None, tmp_path, "data.gold_train")
    with open(uris[0], "a", encoding="utf-8") as fh:
        fh.write("\n")

    with pytest.raises(GoldIOError, match="sha256 mismatch"):
        GoldManifestDataset(uris, meta=meta, repo_root=tmp_path, verify_shards=True)


def test_a_shard_missing_from_the_meta_is_refused_when_verifying(tmp_path: Path):
    uris, meta = _gold_shard_uris_from_cfg(str(_artifact(tmp_path)), None, tmp_path, "data.gold_train")
    other, _ = _gold_shard_uris_from_cfg(str(_artifact(tmp_path / "other")), None, tmp_path, "x")

    with pytest.raises(ValueError, match="not listed in meta"):
        GoldManifestDataset(uris + other, meta=meta, repo_root=tmp_path, verify_shards=True)


class _Captured(Exception):
    pass


def test_the_training_loader_verifies_every_gold_split_it_builds(tmp_path: Path, monkeypatch):
    calls: list[dict] = []

    def record(uris, **kwargs):
        calls.append(kwargs)
        if len(calls) == 3:
            raise _Captured
        return [0]

    monkeypatch.setattr(data_pipeline, "GoldManifestDataset", record)
    metas = {split: str(_artifact(tmp_path, split)) for split in ("train", "val", "test")}
    data = SimpleNamespace(
        pin_memory=False, max_objects=4, condition_mode=None, tag_lighting_metadata=False, batch_size=1,
        shuffle_train=False, num_workers=0, drop_last=False,
        gold_train_meta=metas["train"], gold_train_shard_paths=None,
        gold_val_meta=metas["val"], gold_val_shard_paths=None,
        gold_test_meta=metas["test"], gold_test_shard_paths=None,
    )
    resolved = SimpleNamespace(data=data, model=SimpleNamespace(num_classes=622),
                               training=SimpleNamespace(label_space=LABEL_SPACE_ACCESSIBILITY_622))

    with pytest.raises(_Captured):
        data_pipeline._create_gold_data_loaders_for_resolved(resolved, repo_root=tmp_path, device="cpu")

    assert [c.get("verify_shards") for c in calls] == [True, True, True]
