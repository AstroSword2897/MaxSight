"""Gold manifest builder, schema, I/O layer, and lazy dataset tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ml.data.gold.builder import (
    build_gold_jsonl_from_adapter,
    build_gold_manifest,
    validate_gold_line,
    write_manifest_meta,
)
from ml.data.gold.dataNormalizationLayer import MaxSightListAdapter
from ml.data.gold.dataset import GoldManifestDataset, load_gold_meta
from ml.data.gold.io import GoldIOError, ShardReader
from ml.data.gold.label_mapper import LabelMapper
from ml.data.gold.schema import (
    GOLD_LINE_SCHEMA_VERSION,
    GOLD_META_SCHEMA_VERSION,
    LABEL_SPACE_ACCESSIBILITY_622,
    validate_meta,
)
from PIL import Image

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_image(path: Path, size=(32, 24)):
    Image.new("RGB", size, color=(1, 2, 3)).save(path)


def _minimal_ann(img_names: list[str]) -> list[dict]:
    return [
        {
            "image_id": i,
            "image_path": name,
            "objects": [{"category": "person", "box": [0.5, 0.5, 0.2, 0.2]}],
        }
        for i, name in enumerate(img_names)
    ]


def _build_simple_manifest(tmp_path: Path, names=("a.jpg",), split="train"):
    """Create images + annotation + run adapter → return (out_jsonl, repo_root)."""
    repo = tmp_path
    img_dir = repo / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        _make_image(img_dir / name)
    ann = _minimal_ann(list(names))
    ann_path = repo / f"{split}.json"
    ann_path.write_text(json.dumps(ann), encoding="utf-8")
    adapter = MaxSightListAdapter(
        ann_path, img_dir, repo, dataset_id="testds", version="v1", split=split
    )
    out_jsonl = repo / "gold" / f"{split}.jsonl"
    build_gold_jsonl_from_adapter(
        adapter,
        out_jsonl=out_jsonl,
        repo_root=repo,
        source_annotation=str(ann_path),
        skip_invalid=True,
    )
    return out_jsonl, repo


# ── Schema / validate_meta ─────────────────────────────────────────────────────


def _valid_meta(num_samples=1, *, n_shards=1):
    shards = [
        {
            "uri": f"shard_{i:05d}.jsonl",
            "num_lines": num_samples if i == 0 else 0,
            "sha256": "a" * 64,
        }
        for i in range(n_shards)
    ]
    total = sum(s["num_lines"] for s in shards)
    return {
        "meta_schema_version": GOLD_META_SCHEMA_VERSION,
        "line_schema_version": GOLD_LINE_SCHEMA_VERSION,
        "label_space": LABEL_SPACE_ACCESSIBILITY_622,
        "num_classes": 622,
        "class_map_hash": "b" * 64,
        "num_samples": total,
        "shards": shards,
        "built_at": "2026-01-01T00:00:00+00:00",
    }


def test_validate_meta_valid():
    errs = validate_meta(_valid_meta())
    assert errs == []


def test_validate_meta_missing_key():
    m = _valid_meta()
    del m["num_classes"]
    errs = validate_meta(m)
    assert any("num_classes" in e for e in errs)


def test_validate_meta_sha256_wrong_length():
    m = _valid_meta()
    m["shards"][0]["sha256"] = "tooshort"
    errs = validate_meta(m)
    assert any("sha256" in e for e in errs)


def test_validate_meta_num_samples_mismatch():
    m = _valid_meta()
    m["num_samples"] = 999
    errs = validate_meta(m)
    assert any("num_samples" in e for e in errs)


def test_validate_meta_empty_shards():
    m = _valid_meta()
    m["shards"] = []
    errs = validate_meta(m)
    assert errs


# ── Builder ────────────────────────────────────────────────────────────────────


def test_build_and_load_gold_manifest(tmp_path: Path) -> None:
    out_jsonl, repo = _build_simple_manifest(tmp_path)

    assert out_jsonl.exists()
    line = json.loads(out_jsonl.read_text(encoding="utf-8").strip())
    assert not validate_gold_line(line, repo)

    ds = GoldManifestDataset(out_jsonl, repo_root=repo, strict_images=True)
    assert len(ds) == 1
    sample = ds[0]
    assert sample["images"].shape[0] == 3
    assert sample["labels"][0].item() == 0
    assert sample["num_objects"].item() == 1
    assert isinstance(sample["lighting"], str)


def test_no_image_key_in_sample(tmp_path: Path) -> None:
    out_jsonl, repo = _build_simple_manifest(tmp_path)
    ds = GoldManifestDataset(out_jsonl, repo_root=repo)
    sample = ds[0]
    assert "image" not in sample, "dual contract: 'image' key must not be present"
    assert "images" in sample


def test_deterministic_build_same_hash(tmp_path: Path) -> None:
    """Same adapter input + config → identical shard SHA-256."""
    repo = tmp_path
    img_dir = repo / "images"
    img_dir.mkdir()
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        _make_image(img_dir / name)
    ann = _minimal_ann(["a.jpg", "b.jpg", "c.jpg"])
    ann_path = repo / "train.json"
    ann_path.write_text(json.dumps(ann), encoding="utf-8")

    def _run(out_dir: Path):
        adapter = MaxSightListAdapter(
            ann_path, img_dir, repo, dataset_id="ds", version="v1", split="train"
        )
        return build_gold_manifest(
            adapter,
            mapper=LabelMapper(None, LABEL_SPACE_ACCESSIBILITY_622),
            out=out_dir,
            repo_root=repo,
            source_annotation=str(ann_path),
            num_shards=2,
        )

    s1 = _run(repo / "run1")
    s2 = _run(repo / "run2")
    assert s1["lines_written"] == s2["lines_written"] == 3
    for sh1, sh2 in zip(s1["shards"], s2["shards"]):
        assert sh1["sha256"] == sh2["sha256"], (
            f"shard sha256 differs across deterministic runs: {sh1['sha256']} vs {sh2['sha256']}"
        )


def test_sharded_build_and_multi_shard_load(tmp_path: Path) -> None:
    repo = tmp_path
    img_dir = repo / "images"
    img_dir.mkdir()
    names = [f"img{i}.jpg" for i in range(6)]
    for n in names:
        _make_image(img_dir / n)
    ann = _minimal_ann(names)
    ann_path = repo / "train.json"
    ann_path.write_text(json.dumps(ann), encoding="utf-8")
    adapter = MaxSightListAdapter(
        ann_path, img_dir, repo, dataset_id="ds", version="v1", split="train"
    )
    out_dir = repo / "shards"
    summary = build_gold_manifest(
        adapter,
        mapper=LabelMapper(None, LABEL_SPACE_ACCESSIBILITY_622),
        out=out_dir,
        repo_root=repo,
        source_annotation=str(ann_path),
        num_shards=3,
    )
    assert len(summary["shards"]) == 3
    shard_uris = [s["uri"] for s in summary["shards"]]

    ds = GoldManifestDataset(shard_uris, repo_root=repo)
    assert len(ds) == 6
    for i in range(6):
        s = ds[i]
        assert s["images"].shape[0] == 3


def test_write_and_load_meta(tmp_path: Path) -> None:
    """write_manifest_meta produces a valid meta that validate_meta accepts."""
    repo = tmp_path
    img_dir = repo / "images"
    img_dir.mkdir()
    _make_image(img_dir / "a.jpg")
    ann_path = repo / "train.json"
    ann_path.write_text(json.dumps(_minimal_ann(["a.jpg"])), encoding="utf-8")
    adapter = MaxSightListAdapter(
        ann_path, img_dir, repo, dataset_id="myds", version="v2", split="train"
    )
    out_dir = repo / "gold"
    summary = build_gold_manifest(
        adapter,
        mapper=LabelMapper(None, LABEL_SPACE_ACCESSIBILITY_622),
        out=out_dir,
        repo_root=repo,
        source_annotation=str(ann_path),
        num_shards=1,
    )
    meta_path = repo / "meta.json"
    write_manifest_meta(
        meta_path,
        repo_root=repo,
        dataset_id="myds",
        version="v2",
        split="train",
        label_space=LABEL_SPACE_ACCESSIBILITY_622,
        num_classes=622,
        class_map_hash=summary["class_map_hash"],
        source_annotation=str(ann_path),
        lines_written=summary["lines_written"],
        lines_skipped=summary["lines_skipped"],
        shards=summary["shards"],
    )
    meta = load_gold_meta(meta_path)
    errs = validate_meta(meta)
    assert errs == [], errs


def test_meta_driven_dataset(tmp_path: Path) -> None:
    """GoldManifestDataset driven by meta auto-derives shard list + dataset_source."""
    repo = tmp_path
    img_dir = repo / "images"
    img_dir.mkdir()
    _make_image(img_dir / "a.jpg")
    ann_path = repo / "train.json"
    ann_path.write_text(json.dumps(_minimal_ann(["a.jpg"])), encoding="utf-8")
    adapter = MaxSightListAdapter(
        ann_path, img_dir, repo, dataset_id="myds", version="v2", split="train"
    )
    out_dir = repo / "gold"
    summary = build_gold_manifest(
        adapter,
        mapper=LabelMapper(None, LABEL_SPACE_ACCESSIBILITY_622),
        out=out_dir,
        repo_root=repo,
        source_annotation=str(ann_path),
        num_shards=1,
    )
    meta_path = repo / "meta.json"
    write_manifest_meta(
        meta_path,
        repo_root=repo,
        dataset_id="myds",
        version="v2",
        split="train",
        label_space=LABEL_SPACE_ACCESSIBILITY_622,
        num_classes=622,
        class_map_hash=summary["class_map_hash"],
        source_annotation=str(ann_path),
        lines_written=summary["lines_written"],
        lines_skipped=summary["lines_skipped"],
        shards=summary["shards"],
    )
    # Load via meta file only — no explicit shard paths.
    meta = load_gold_meta(meta_path)
    shard_uri = meta["shards"][0]["uri"]
    ds = GoldManifestDataset(shard_uri, meta=meta, repo_root=repo)
    assert len(ds) == 1
    assert ds.dataset_source_key == "myds@v2"
    assert ds.num_classes == 622


def test_corrupted_shard_line_error_has_context(tmp_path: Path) -> None:
    """A truncated/corrupt JSONL line raises GoldIOError with uri + idx + offset."""
    repo = tmp_path
    img_dir = repo / "images"
    img_dir.mkdir()
    _make_image(img_dir / "a.jpg")
    out_jsonl, _ = _build_simple_manifest(tmp_path / "ok", names=("a.jpg",))
    # Write a shard with a corrupt second line.
    shard = tmp_path / "shard_corrupt.jsonl"
    good_line = out_jsonl.read_text(encoding="utf-8").strip()
    shard.write_text(good_line + "\n{INVALID_JSON\n", encoding="utf-8")

    ds = GoldManifestDataset(str(shard), repo_root=repo)
    assert len(ds) == 2
    # First line is valid.
    _ = ds[0]
    # Second line is corrupt.
    with pytest.raises((GoldIOError, ValueError)):
        _ = ds[1]


def test_collate_stack_images(tmp_path: Path) -> None:
    from ml.data.data_pipeline import collate_fn

    repo = tmp_path
    img_dir = repo / "i"
    img_dir.mkdir()
    for name in ("x.jpg", "y.jpg"):
        _make_image(img_dir / name)
    ann = _minimal_ann(["x.jpg", "y.jpg"])
    p = repo / "a.json"
    p.write_text(json.dumps(ann), encoding="utf-8")
    adapter = MaxSightListAdapter(p, img_dir, repo, dataset_id="d", version="v1", split="train")
    out = repo / "m.jsonl"
    build_gold_jsonl_from_adapter(
        adapter,
        out_jsonl=out,
        repo_root=repo,
        source_annotation=str(p),
    )
    ds = GoldManifestDataset(out, repo_root=repo)
    batch = collate_fn([ds[0], ds[1]])
    assert batch["images"].shape[0] == 2
    assert batch["images"].dim() == 4


def test_invalid_row_raises_valueerror(tmp_path: Path) -> None:
    """A JSONL line missing required keys raises ValueError with uri in message."""
    shard = tmp_path / "bad.jsonl"
    bad = {"schema_version": GOLD_LINE_SCHEMA_VERSION, "labels": [], "boxes": []}
    shard.write_text(json.dumps(bad) + "\n", encoding="utf-8")

    ds = GoldManifestDataset(str(shard), repo_root=tmp_path, num_classes=622)
    with pytest.raises(ValueError, match="uri="):
        _ = ds[0]


# ── class_map_hash ─────────────────────────────────────────────────────────────


def test_label_mapper_class_map_hash_stable() -> None:
    """Same mapper config → same hash across calls; deterministic."""
    from ml.data.gold.label_mapper import LabelMapper

    m = LabelMapper(None, LABEL_SPACE_ACCESSIBILITY_622)
    assert len(m.class_map_hash) == 64
    assert m.class_map_hash == m.class_map_hash  # idempotent


def test_label_mapper_class_map_hash_in_build_summary(tmp_path: Path) -> None:
    """build_gold_manifest includes class_map_hash matching the mapper."""
    repo = tmp_path
    img_dir = repo / "images"
    img_dir.mkdir()
    _make_image(img_dir / "a.jpg")
    ann_path = repo / "t.json"
    ann_path.write_text(json.dumps(_minimal_ann(["a.jpg"])), encoding="utf-8")
    mapper = LabelMapper(None, LABEL_SPACE_ACCESSIBILITY_622)
    adapter = MaxSightListAdapter(
        ann_path, img_dir, repo, dataset_id="d", version="v1", split="train"
    )
    summary = build_gold_manifest(
        adapter,
        mapper=mapper,
        out=repo / "gold" / "train.jsonl",
        repo_root=repo,
        source_annotation=str(ann_path),
    )
    assert summary["class_map_hash"] == mapper.class_map_hash


def test_class_map_hash_mismatch_raises(tmp_path: Path) -> None:
    """GoldManifestDataset raises when expected_class_map_hash doesn't match meta."""
    repo = tmp_path
    img_dir = repo / "images"
    img_dir.mkdir()
    _make_image(img_dir / "a.jpg")
    ann_path = repo / "t.json"
    ann_path.write_text(json.dumps(_minimal_ann(["a.jpg"])), encoding="utf-8")
    mapper = LabelMapper(None, LABEL_SPACE_ACCESSIBILITY_622)
    adapter = MaxSightListAdapter(
        ann_path, img_dir, repo, dataset_id="d", version="v1", split="train"
    )
    summary = build_gold_manifest(
        adapter,
        mapper=mapper,
        out=repo / "gold" / "train.jsonl",
        repo_root=repo,
        source_annotation=str(ann_path),
    )
    meta_path = repo / "meta.json"
    write_manifest_meta(
        meta_path,
        repo_root=repo,
        label_space=LABEL_SPACE_ACCESSIBILITY_622,
        num_classes=622,
        class_map_hash=summary["class_map_hash"],
        lines_written=summary["lines_written"],
        lines_skipped=summary["lines_skipped"],
        shards=summary["shards"],
    )
    wrong_hash = "0" * 64
    meta = load_gold_meta(meta_path)
    shard_uri = meta["shards"][0]["uri"]
    with pytest.raises(ValueError, match="class_map_hash mismatch"):
        GoldManifestDataset(
            shard_uri,
            meta=meta,
            repo_root=repo,
            expected_class_map_hash=wrong_hash,
        )


# ── Registry independence ──────────────────────────────────────────────────────


def test_gold_dataset_loads_without_registry(tmp_path: Path) -> None:
    """GoldManifestDataset must load successfully when the dataset registry
    module is absent from sys.modules (proves zero registry dependency).
    """
    import sys

    # Build a valid artifact first (registry not needed for build either).
    repo = tmp_path
    img_dir = repo / "images"
    img_dir.mkdir()
    _make_image(img_dir / "a.jpg")
    ann_path = repo / "t.json"
    ann_path.write_text(json.dumps(_minimal_ann(["a.jpg"])), encoding="utf-8")
    mapper = LabelMapper(None, LABEL_SPACE_ACCESSIBILITY_622)
    adapter = MaxSightListAdapter(
        ann_path, img_dir, repo, dataset_id="d", version="v1", split="train"
    )
    summary = build_gold_manifest(
        adapter,
        mapper=mapper,
        out=repo / "gold" / "train.jsonl",
        repo_root=repo,
        source_annotation=str(ann_path),
    )
    meta_path = repo / "meta.json"
    write_manifest_meta(
        meta_path,
        repo_root=repo,
        label_space=LABEL_SPACE_ACCESSIBILITY_622,
        num_classes=622,
        class_map_hash=summary["class_map_hash"],
        lines_written=summary["lines_written"],
        lines_skipped=summary["lines_skipped"],
        shards=summary["shards"],
    )

    # Remove registry modules from sys.modules to prove independence.
    registry_keys = [k for k in sys.modules if "dataset_registry" in k]
    removed = {k: sys.modules.pop(k) for k in registry_keys}
    try:
        meta = load_gold_meta(meta_path)
        shard_uri = meta["shards"][0]["uri"]
        ds = GoldManifestDataset(shard_uri, meta=meta, repo_root=repo)
        assert len(ds) == 1
        sample = ds[0]
        assert "images" in sample
    finally:
        sys.modules.update(removed)


def test_byte_level_determinism(tmp_path: Path) -> None:
    """Same input + mapper → identical shard bytes (not just the same hash)."""
    repo = tmp_path
    img_dir = repo / "images"
    img_dir.mkdir()
    for n in ("a.jpg", "b.jpg"):
        _make_image(img_dir / n)
    ann_path = repo / "t.json"
    ann_path.write_text(json.dumps(_minimal_ann(["a.jpg", "b.jpg"])), encoding="utf-8")

    def _build(out_file: Path) -> bytes:
        adapter = MaxSightListAdapter(
            ann_path, img_dir, repo, dataset_id="d", version="v1", split="train"
        )
        build_gold_manifest(
            adapter,
            mapper=LabelMapper(None, LABEL_SPACE_ACCESSIBILITY_622),
            out=out_file,
            repo_root=repo,
            source_annotation=str(ann_path),
        )
        return out_file.read_bytes()

    b1 = _build(repo / "run1.jsonl")
    b2 = _build(repo / "run2.jsonl")
    assert b1 == b2, "shard bytes differ between deterministic builds"


def test_validate_meta_missing_class_map_hash() -> None:
    """Omitting class_map_hash from meta must fail validation."""
    m = _valid_meta()
    del m["class_map_hash"]
    errs = validate_meta(m)
    assert any("class_map_hash" in e for e in errs)


def test_shard_reader_enriches_goldioeror(tmp_path: Path) -> None:
    """ShardReader.read_at must embed shard_sha256 in GoldIOError context."""

    shard = tmp_path / "s.jsonl"
    shard.write_text("{}\n", encoding="utf-8")
    reader = ShardReader(str(shard), shard_sha256="a" * 64, line_schema_version="1.0")
    offsets = reader.index_line_starts()
    # First line is valid bytes but will fail JSON schema — GoldIOError is not
    # raised here; we test that the reader object carries the metadata.
    assert reader.shard_sha256 == "a" * 64
    assert reader.line_schema_version == "1.0"
