"""Write validated gold JSONL (optionally sharded) plus reproducibility sidecars."""

from __future__ import annotations

import hashlib
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ml.data.gold.label_mapper import LabelMapper
from ml.data.gold.schema import (
    GOLD_LINE_SCHEMA_VERSION,
    GOLD_META_SCHEMA_VERSION,
    LABEL_SPACE_ACCESSIBILITY_622,
    REQUIRED_LINE_KEYS,
)
from ml.data.medallion_layout import resolve_repo_path
logger = logging.getLogger(__name__)

_METADATA_REQUIRED = frozenset(
    {
        "dataset_id",
        "split",
        "label_space",
        "source_file",
        "original_id",
        "width",
        "height",
    }
)


def finalize_gold_record(partial: Dict[str, Any], mapper: LabelMapper) -> Dict[str, Any]:
    """Apply label mapping; partial rows must contain ``label_names`` not ``labels``."""

    if "labels" in partial:
        raise ValueError("finalize_gold_record: partial must not contain 'labels' yet")
    names = partial.get("label_names")
    if not isinstance(names, list):
        raise ValueError("finalize_gold_record: partial must contain label_names list")
    rec = {k: v for k, v in partial.items() if k != "label_names"}
    rec["labels"] = mapper.map_class_names(names)
    return rec


def _stable_sort_key(record: Dict[str, Any]) -> Tuple[str, str, str]:
    meta = record.get("metadata") or {}
    return (
        str(record.get("image_path", "")),
        str(meta.get("original_id", "")),
        str(meta.get("source_file", "")),
    )


def _shard_ranges(n: int, num_shards: int) -> List[Tuple[int, int]]:
    if n == 0:
        return [(0, 0)] * max(1, num_shards)
    if num_shards <= 1:
        return [(0, n)]
    ranges: List[Tuple[int, int]] = []
    base = n // num_shards
    rem = n % num_shards
    start = 0
    for i in range(num_shards):
        extra = 1 if i < rem else 0
        end = start + base + extra
        ranges.append((start, end))
        start = end
    return ranges


def validate_gold_line_in_memory(
    record: Dict[str, Any],
    *,
    expected_label_space: str,
    num_classes: int = 0,
) -> List[str]:
    """Schema checks without touching disk (for DataLoader hot path)."""

    errs: List[str] = []
    if not isinstance(record, dict):
        return ["record root must be an object"]
    missing = sorted(REQUIRED_LINE_KEYS - record.keys())
    if missing:
        errs.append(f"missing keys: {missing}")
        return errs
    if record.get("schema_version") != GOLD_LINE_SCHEMA_VERSION:
        errs.append(
            f"schema_version: expected {GOLD_LINE_SCHEMA_VERSION!r}, got {record.get('schema_version')!r}"
        )
    meta = record.get("metadata")
    if not isinstance(meta, dict):
        errs.append("metadata must be an object")
        return errs
    miss_meta = sorted(_METADATA_REQUIRED - set(meta.keys()))
    if miss_meta:
        errs.append(f"metadata missing keys: {miss_meta}")
        return errs
    for key in _METADATA_REQUIRED:
        val = meta[key]
        if key in ("width", "height"):
            if type(val) is not int or isinstance(val, bool) or val < 0:
                errs.append(f"metadata.{key} must be a non-negative int")
        elif key == "original_id":
            if not (isinstance(val, (str, int)) and str(val).strip()):
                errs.append("metadata.original_id must be a non-empty str or int")
        elif key == "label_space":
            if not isinstance(val, str) or not val.strip():
                errs.append("metadata.label_space must be a non-empty string")
        elif not isinstance(val, str) or not val.strip():
            errs.append(f"metadata.{key} must be a non-empty string")
    if meta.get("label_space") != expected_label_space:
        errs.append(
            f"metadata.label_space: expected {expected_label_space!r}, got {meta.get('label_space')!r}"
        )
    boxes = record.get("boxes")
    labels = record.get("labels")
    if not isinstance(boxes, list) or not isinstance(labels, list):
        errs.append("boxes and labels must be arrays")
        return errs
    if len(boxes) != len(labels):
        errs.append(f"boxes length {len(boxes)} != labels length {len(labels)}")
    dists = record.get("distances")
    if dists is not None and (
        not isinstance(dists, list) or len(dists) != len(labels)
    ):
        errs.append("distances must be same length as labels when present")
    ou = record.get("object_urgencies")
    if ou is not None and (not isinstance(ou, list) or len(ou) != len(labels)):
        errs.append("object_urgencies must be same length as labels when present")
    ip = record.get("image_path")
    if not isinstance(ip, str) or not ip.strip():
        errs.append("image_path invalid")
    for i, lab in enumerate(labels):
        if not isinstance(lab, int) or lab < 0 or (num_classes > 0 and lab >= num_classes):
            errs.append(f"labels[{i}] invalid (num_classes={num_classes}): {lab!r}")
    for i, b in enumerate(boxes):
        if (
            not isinstance(b, (list, tuple))
            or len(b) != 4
            or not all(isinstance(x, (int, float)) for x in b)
        ):
            errs.append(f"boxes[{i}] must be length-4 numeric")
            continue
        for j, x in enumerate(b):
            if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
                errs.append(f"boxes[{i}][{j}] is nan/inf")
        if len(b) == 4:
            cx, cy, w, h = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
                errs.append(f"boxes[{i}] center out of [0,1]")
            if w < 1e-4 or h < 1e-4 or w > 1.0 or h > 1.0:
                errs.append(f"boxes[{i}] size out of allowed range")
    return errs


def validate_gold_line(
    record: Dict[str, Any],
    repo_root: Path,
    *,
    expected_label_space: str = LABEL_SPACE_ACCESSIBILITY_622,
) -> List[str]:
    """Full row validation including that the image file resolves on disk."""

    errs = validate_gold_line_in_memory(
        record, expected_label_space=expected_label_space
    )
    if errs:
        return errs
    ip = record.get("image_path")
    if isinstance(ip, str) and ip.strip():
        abs_p = resolve_repo_path(Path(repo_root), ip)
        if not abs_p.is_file():
            errs.append(f"image_path not found: {abs_p}")
    return errs


def _dumps_line(record: Dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_gold_manifest(
    adapter: Any,
    *,
    mapper: LabelMapper,
    out: Path,
    repo_root: Path,
    source_annotation: str,
    num_shards: int = 1,
    skip_invalid: bool = True,
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Collect, sort, map labels, validate, write shard JSONL files.

    When ``num_shards`` is 1, ``out`` is the output ``.jsonl`` file path.
    When ``num_shards`` > 1, ``out`` is a directory; writes ``shard_00000.jsonl``, …

    The returned summary includes ``class_map_hash`` so callers can embed it
    in meta.json without recomputing it separately.
    """

    def _log(msg: str) -> None:
        if log:
            log(msg)
        else:
            logger.info("%s", msg)

    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    rr = Path(repo_root).resolve()
    if not hasattr(adapter, "load_partial"):
        raise TypeError("adapter must implement load_partial(idx)")
    n = len(adapter)
    collected: List[Dict[str, Any]] = []
    skipped = 0
    for idx in range(n):
        partial = adapter.load_partial(idx)
        record = finalize_gold_record(partial, mapper)
        errs = validate_gold_line(
            record, rr, expected_label_space=mapper.target_space
        )
        if errs:
            skipped += 1
            msg = f"skip idx={idx}: " + "; ".join(errs)
            if skip_invalid:
                _log(msg)
                continue
            raise ValueError(msg)
        collected.append(record)

    collected.sort(key=_stable_sort_key)
    cmh = mapper.class_map_hash

    def _write_shard(records: List[Dict[str, Any]], path: Path) -> str:
        """Write records to ``path`` and return its SHA-256.

        Encodes as UTF-8 with LF line endings — no CRLF; no BOM.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        h = hashlib.sha256()
        with path.open("wb") as f:
            for rec in records:
                # Always LF (\n), never CRLF; encoding enforced at byte level.
                line_bytes = (_dumps_line(rec) + "\n").encode("utf-8")
                assert b"\r" not in line_bytes, "CRLF must not appear in gold shards"
                f.write(line_bytes)
                h.update(line_bytes)
        return h.hexdigest()

    if num_shards == 1:
        out_path = Path(out)
        sha = _write_shard(collected, out_path)
        return {
            "lines_written": len(collected),
            "lines_skipped": skipped,
            "class_map_hash": cmh,
            "shards": [
                {
                    "uri": str(out_path.resolve()),
                    "num_lines": len(collected),
                    "sha256": sha,
                }
            ],
        }

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ranges = _shard_ranges(len(collected), num_shards)
    shard_infos: List[Dict[str, Any]] = []
    for si, (lo, hi) in enumerate(ranges):
        shard_path = out_dir / f"shard_{si:05d}.jsonl"
        chunk = collected[lo:hi]
        sha = _write_shard(chunk, shard_path)
        shard_infos.append(
            {
                "uri": str(shard_path.resolve()),
                "num_lines": len(chunk),
                "sha256": sha,
            }
        )
    return {
        "lines_written": len(collected),
        "lines_skipped": skipped,
        "class_map_hash": cmh,
        "shards": shard_infos,
    }


def build_gold_jsonl_from_adapter(
    adapter: Any,
    *,
    out_jsonl: Path,
    repo_root: Path,
    source_annotation: str,
    label_space: str = LABEL_SPACE_ACCESSIBILITY_622,
    source_label_space: Optional[str] = None,
    num_shards: int = 1,
    skip_invalid: bool = True,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[int, int, str]:
    """Backward-compatible wrapper returning ``(written, skipped, sha256)`` for shard 0."""

    mapper = LabelMapper(source_label_space, label_space)
    summary = build_gold_manifest(
        adapter,
        mapper=mapper,
        out=out_jsonl,
        repo_root=repo_root,
        source_annotation=source_annotation,
        num_shards=num_shards,
        skip_invalid=skip_invalid,
        log=log,
    )
    sha0 = summary["shards"][0]["sha256"] if summary["shards"] else ""
    return summary["lines_written"], summary["lines_skipped"], sha0


def write_manifest_meta(
    meta_path: Path,
    *,
    repo_root: Path,
    label_space: str,
    num_classes: int,
    class_map_hash: str,
    lines_written: int,
    lines_skipped: int,
    shards: Sequence[Dict[str, Any]],
    # ── Provenance (optional; never used for runtime branching) ────────────────
    dataset_id: str = "",
    version: str = "",
    split: str = "",
    source_annotation: str = "",
) -> None:
    """Write portable gold artifact meta (meta.json).

    The file is the single source of truth for the artifact: all runtime-
    critical fields (label_space, num_classes, class_map_hash, shards) are
    required.  Provenance fields (dataset_id, version, …) are optional and
    written only for human debugging.
    """
    meta_path = Path(meta_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "meta_schema_version": GOLD_META_SCHEMA_VERSION,
        "line_schema_version": GOLD_LINE_SCHEMA_VERSION,
        "label_space": label_space,
        "num_classes": int(num_classes),
        "class_map_hash": class_map_hash,
        "num_samples": lines_written,
        "shards": list(shards),
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    # Provenance — only written when provided.
    for k, v in [
        ("dataset_id", dataset_id),
        ("version", version),
        ("split", split),
        ("source_annotation", source_annotation),
        ("num_skipped", lines_skipped),
        ("repo_root", str(Path(repo_root).resolve())),
    ]:
        if v not in ("", 0):
            payload[k] = v
    meta_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
