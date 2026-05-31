"""Constants for gold manifest lines (JSONL) and the portable artifact meta contract.

Artifact layout
---------------
A gold artifact consists of:
  meta.json          — required; self-describing; portable (no repo_root dependency)
  shard_00000.jsonl  — one or more JSONL shard files
  shard_00001.jsonl
  ...

``meta.json`` makes the artifact self-sufficient: training, validation, and
inference can resolve every invariant (label_space, num_classes, shard uris,
integrity hashes) from the meta alone — no dataset registry needed at runtime.
"""

from typing import Any

# ── JSONL row schema ───────────────────────────────────────────────────────────

GOLD_LINE_SCHEMA_VERSION = "1.0"
LABEL_SPACE_ACCESSIBILITY_622 = "accessibility_622"

REQUIRED_LINE_KEYS = frozenset({"schema_version", "image_path", "boxes", "labels", "metadata"})

# ── Artifact meta schema ───────────────────────────────────────────────────────

GOLD_META_SCHEMA_VERSION = "1.0"

# ── Required meta keys (runtime-critical; absence must abort loading) ──────────
# The runtime must NEVER branch on provenance-only keys (dataset_id, version,
# split) — they are informational.  Only the fields below gate correctness.
REQUIRED_META_KEYS = frozenset(
    {
        "meta_schema_version",
        "line_schema_version",
        "label_space",
        "num_classes",
        "class_map_hash",  # SHA-256 over ordered (idx, name) pairs — guards label drift
        "num_samples",
        "shards",
        "built_at",
    }
)

# ── Optional provenance keys (never used for runtime branching) ────────────────
PROVENANCE_META_KEYS = frozenset(
    {
        "dataset_id",
        "version",
        "split",
        "source_datasets",
        "source_annotation",
        "repo_root",
        "num_skipped",
    }
)

# Required keys in each shard entry within meta["shards"]
REQUIRED_SHARD_ENTRY_KEYS = frozenset({"uri", "num_lines", "sha256"})


def validate_meta(meta: dict[str, Any]) -> list[str]:
    """Return human-readable errors for a meta dict; empty list means valid.

    Purely in-memory — does NOT check whether shard URIs are reachable.
    Unknown keys that are in PROVENANCE_META_KEYS are allowed without error;
    any other unknown key is flagged to catch typos early.
    """
    if not isinstance(meta, dict):
        return ["meta root must be a JSON object"]
    errs: list[str] = []
    missing = sorted(REQUIRED_META_KEYS - meta.keys())
    if missing:
        errs.append(f"meta missing required keys: {missing}")
        return errs
    unknown = set(meta.keys()) - REQUIRED_META_KEYS - PROVENANCE_META_KEYS
    if unknown:
        errs.append(f"meta has unexpected keys: {sorted(unknown)}")
    if meta.get("meta_schema_version") != GOLD_META_SCHEMA_VERSION:
        errs.append(
            f"meta_schema_version: expected {GOLD_META_SCHEMA_VERSION!r}, "
            f"got {meta.get('meta_schema_version')!r}"
        )
    if meta.get("line_schema_version") != GOLD_LINE_SCHEMA_VERSION:
        errs.append(
            f"line_schema_version: expected {GOLD_LINE_SCHEMA_VERSION!r}, "
            f"got {meta.get('line_schema_version')!r}"
        )
    ls = meta.get("label_space")
    if not isinstance(ls, str) or not ls.strip():
        errs.append("meta.label_space must be a non-empty string")
    nc = meta.get("num_classes")
    if not isinstance(nc, int) or nc <= 0:
        errs.append("meta.num_classes must be a positive integer")
    cmh = meta.get("class_map_hash")
    if not isinstance(cmh, str) or len(cmh) != 64:
        errs.append("meta.class_map_hash must be a 64-char hex SHA-256 string")
    ns = meta.get("num_samples")
    if not isinstance(ns, int) or ns < 0:
        errs.append("meta.num_samples must be a non-negative integer")
    shards = meta.get("shards")
    if not isinstance(shards, list) or not shards:
        errs.append("meta.shards must be a non-empty list")
    else:
        total_lines = 0
        for i, s in enumerate(shards):
            if not isinstance(s, dict):
                errs.append(f"meta.shards[{i}] must be an object")
                continue
            sm = sorted(REQUIRED_SHARD_ENTRY_KEYS - s.keys())
            if sm:
                errs.append(f"meta.shards[{i}] missing keys: {sm}")
                continue
            if not isinstance(s.get("uri"), str) or not s["uri"].strip():
                errs.append(f"meta.shards[{i}].uri must be a non-empty string")
            nl = s.get("num_lines")
            if not isinstance(nl, int) or nl < 0:
                errs.append(f"meta.shards[{i}].num_lines must be a non-negative int")
            else:
                total_lines += nl
            sha = s.get("sha256")
            if not isinstance(sha, str) or len(sha) != 64:
                errs.append(f"meta.shards[{i}].sha256 must be a 64-char hex string")
        if not errs:
            total_in_meta = int(meta.get("num_samples", 0))
            if total_lines != total_in_meta:
                errs.append(
                    f"meta.num_samples ({total_in_meta}) != sum of shard num_lines ({total_lines})"
                )
    return errs
