"""Lazy torch Dataset over one or more gold JSONL shard files.

Accepts local paths or ``s3://`` URIs for every shard.  When a meta dict/path
is supplied the dataset is fully self-describing: num_classes, label_space, and
dataset_source_key are derived from the artifact meta rather than from the
registry.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ml.data.assistive_supervision import load_assistive_spec, object_distance_and_urgency
from ml.data.gold.builder import validate_gold_line_in_memory
from ml.data.gold.io import GoldIOError, ShardReader
from ml.data.gold.schema import (
    GOLD_LINE_SCHEMA_VERSION,
    LABEL_SPACE_ACCESSIBILITY_622,
    validate_meta,
)
from ml.models.maxsight_cnn import COCO_CLASSES
from ml.utils.preprocessing import ImagePreprocessor

logger = logging.getLogger(__name__)

_GOLD_ASSISTIVE_SPEC = load_assistive_spec()

# Shard URIs are strings; may be absolute local paths or s3:// URIs.
_URI = str


def _flatten_shard_index(
    readers: Sequence["ShardReader"],
) -> List[Tuple["ShardReader", int]]:
    """Build a flat ``(ShardReader, byte_offset)`` index across all shards."""
    flat: List[Tuple[ShardReader, int]] = []
    for reader in readers:
        for off in reader.index_line_starts():
            flat.append((reader, off))
    return flat


def _resolve_image_path(image_path: str, repo_root: Optional[Path]) -> Path:
    """Resolve a (possibly relative) image path against repo_root when needed."""
    p = Path(image_path)
    if p.is_absolute():
        return p
    if repo_root is not None:
        return repo_root / p
    # Best-effort: treat as relative to cwd.
    return p


def load_gold_meta(meta_path: Union[str, Path]) -> Dict[str, Any]:
    """Load and validate a gold artifact meta.json; raise on schema errors."""
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    errs = validate_meta(meta)
    if errs:
        raise ValueError(
            f"Invalid gold meta at {meta_path}: " + "; ".join(errs)
        )
    return meta


class GoldManifestDataset(Dataset):
    """Validated gold lines; ``images`` tensor matches the MaxSight collate contract.

    Parameters
    ----------
    shard_uris:
        One or more local paths or ``s3://`` URIs pointing to gold JSONL shards.
        Accepts a single ``str``/``Path`` or a sequence of them.
    meta:
        Optional gold artifact meta dict (already loaded) or path to ``meta.json``.
        When supplied, ``num_classes``, ``label_space``, ``class_map_hash``, and
        ``dataset_source_key`` are derived from the artifact — no registry lookup
        is required.
    repo_root:
        Base directory used to resolve relative ``image_path`` values.  Not
        required when all image paths are absolute or S3-backed.
    expected_class_map_hash:
        When provided, the meta's ``class_map_hash`` must match exactly.  Use
        this to catch silent label-ordering drift between build and load time.
    verify_shards:
        When ``True``, SHA-256 of every shard is verified against the meta
        before indexing.  Fails fast on corruption; adds startup I/O cost.
    """

    def __init__(
        self,
        shard_uris: Union[str, Path, Sequence[Union[str, Path]]],
        *,
        meta: Optional[Union[Dict[str, Any], str, Path]] = None,
        repo_root: Optional[Path] = None,
        max_objects: int = 10,
        condition_mode: Optional[str] = None,
        tag_lighting_metadata: bool = False,
        strict_images: bool = True,
        dataset_source_key: str = "",
        expected_label_space: str = LABEL_SPACE_ACCESSIBILITY_622,
        num_classes: int = 0,
        expected_class_map_hash: Optional[str] = None,
        verify_shards: bool = False,
    ) -> None:
        # Normalise shard_uris → tuple of URI strings.
        if isinstance(shard_uris, (str, Path)):
            raw_seq: Sequence[Union[str, Path]] = (shard_uris,)
        else:
            raw_seq = shard_uris
        uris: Tuple[_URI, ...] = tuple(str(u) for u in raw_seq)
        if not uris:
            raise ValueError("GoldManifestDataset requires at least one shard URI")

        # Load + validate meta when given.
        self._meta: Optional[Dict[str, Any]] = None
        if meta is not None:
            if isinstance(meta, (str, Path)):
                self._meta = load_gold_meta(meta)
            else:
                errs = validate_meta(meta)
                if errs:
                    raise ValueError(
                        "GoldManifestDataset: invalid meta dict: " + "; ".join(errs)
                    )
                self._meta = dict(meta)

        # Derive dataset identity from meta when present; callers can override.
        if self._meta is not None:
            if not dataset_source_key:
                mid = self._meta.get("dataset_id", "")
                mver = self._meta.get("version", "")
                dataset_source_key = f"{mid}@{mver}" if mver else mid
            if not expected_label_space or expected_label_space == LABEL_SPACE_ACCESSIBILITY_622:
                expected_label_space = self._meta.get("label_space", expected_label_space)
            if num_classes == 0:
                num_classes = int(self._meta.get("num_classes", 0))
            # Verify class_map_hash when caller provides an expectation.
            meta_cmh = self._meta.get("class_map_hash", "")
            if expected_class_map_hash and meta_cmh != expected_class_map_hash:
                raise ValueError(
                    f"GoldManifestDataset: class_map_hash mismatch — "
                    f"expected {expected_class_map_hash!r}, meta has {meta_cmh!r}. "
                    "The artifact was built with a different class ordering."
                )

        self.repo_root = Path(repo_root).resolve() if repo_root else None
        self.max_objects = max_objects
        self.strict_images = strict_images
        self.dataset_source_key = dataset_source_key or ""
        self.tag_lighting_metadata = tag_lighting_metadata
        self.preprocessor = ImagePreprocessor(condition_mode=condition_mode)
        self.expected_label_space = expected_label_space
        self.num_classes = int(num_classes)

        # Build per-shard SHA-256 lookup from meta for enriched error messages.
        shard_sha_map: Dict[str, str] = {}
        if self._meta:
            for s in self._meta.get("shards", []):
                if "uri" in s and "sha256" in s:
                    shard_sha_map[s["uri"]] = s["sha256"]

        self._readers: Tuple[ShardReader, ...] = tuple(
            ShardReader(
                uri,
                shard_sha256=shard_sha_map.get(uri),
                line_schema_version=GOLD_LINE_SCHEMA_VERSION,
            )
            for uri in uris
        )

        if verify_shards and self._meta:
            self._verify_shard_integrity(shard_sha_map)

        self._index: List[Tuple[ShardReader, int]] = _flatten_shard_index(self._readers)
        self._missing_warned = False

    def _verify_shard_integrity(self, sha_map: Dict[str, str]) -> None:
        """Re-hash every shard against meta sha256; raises ``GoldIOError`` on mismatch."""
        for reader in self._readers:
            expected = sha_map.get(reader.uri)
            if expected:
                reader.verify_sha256(expected)
            else:
                logger.warning(
                    "GoldManifestDataset: no sha256 in meta for shard %s; skipping verification",
                    reader.uri,
                )

    def __len__(self) -> int:
        return len(self._index)

    def _read_line_dict(self, idx: int) -> Dict[str, Any]:
        reader, off = self._index[idx]
        raw = reader.read_at(idx, off)
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GoldIOError(
                uri=reader.uri,
                idx=idx,
                offset=off,
                raw_prefix=raw[:80],
                reason=f"{type(exc).__name__}: {exc}",
                shard_sha256=reader.shard_sha256,
                line_schema_version=reader.line_schema_version,
            ) from exc

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self._read_line_dict(idx)

        if record.get("schema_version") != GOLD_LINE_SCHEMA_VERSION:
            reader, off = self._index[idx]
            raise ValueError(
                f"GoldManifestDataset: unsupported schema_version "
                f"{record.get('schema_version')!r} | "
                f"uri={reader.uri!r} idx={idx} offset={off}"
            )

        mem_errs = validate_gold_line_in_memory(
            record,
            expected_label_space=self.expected_label_space,
            num_classes=self.num_classes,
        )
        if mem_errs:
            reader, off = self._index[idx]
            raise ValueError(
                f"GoldManifestDataset: invalid row | "
                f"uri={reader.uri!r} idx={idx} offset={off} | "
                + "; ".join(mem_errs)
            )

        reader, _off = self._index[idx]
        image_path = record["image_path"]
        abs_img = _resolve_image_path(image_path, self.repo_root)

        if not abs_img.exists():
            if self.strict_images:
                raise FileNotFoundError(
                    f"GoldManifestDataset strict_images=True: missing {abs_img}"
                )
            if not self._missing_warned:
                logger.warning(
                    "GoldManifestDataset: missing image (random fill). path=%s", abs_img
                )
                self._missing_warned = True
            image = Image.fromarray(
                np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            )
        else:
            try:
                image = Image.open(abs_img).convert("RGB")
            except Exception as exc:
                if self.strict_images:
                    raise RuntimeError(
                        f"GoldManifestDataset strict_images=True: decode failed {abs_img}: {exc}"
                    ) from exc
                if not self._missing_warned:
                    logger.warning(
                        "GoldManifestDataset: decode failed (random fill). path=%s err=%s",
                        abs_img,
                        exc,
                    )
                    self._missing_warned = True
                image = Image.fromarray(
                    np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                )

        meta = record.get("metadata") or {}
        conditions = meta.get("conditions") or {}
        lighting = conditions.get("lighting", "normal")

        if self.tag_lighting_metadata:
            preprocessed = self.preprocessor.preprocess_with_lighting(image)
            image_tensor = preprocessed["image"]
            lighting = preprocessed.get("lighting", lighting)
        else:
            image_tensor = self.preprocessor(image)

        raw_labels = record.get("labels") or []
        raw_boxes = record.get("boxes") or []
        raw_dist = record.get("distances")
        num_objs = min(len(raw_labels), len(raw_boxes), self.max_objects)

        labels = torch.zeros(self.max_objects, dtype=torch.long)
        boxes = torch.zeros(self.max_objects, 4, dtype=torch.float32)
        distance = torch.zeros(self.max_objects, dtype=torch.long)

        for i in range(num_objs):
            labels[i] = int(raw_labels[i])
            b = raw_boxes[i]
            box_tensor = torch.tensor(
                [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                dtype=torch.float32,
            )
            box_tensor[0] = torch.clamp(box_tensor[0], 0.0, 1.0)
            box_tensor[1] = torch.clamp(box_tensor[1], 0.0, 1.0)
            box_tensor[2] = torch.clamp(box_tensor[2], 1e-4, 1.0)
            box_tensor[3] = torch.clamp(box_tensor[3], 1e-4, 1.0)
            if torch.isnan(box_tensor).any() or torch.isinf(box_tensor).any():
                box_tensor = torch.tensor([0.5, 0.5, 0.1, 0.1], dtype=torch.float32)
            boxes[i] = box_tensor
            if isinstance(raw_dist, list) and i < len(raw_dist):
                distance[i] = int(raw_dist[i])
            else:
                li = int(labels[i])
                cat = COCO_CLASSES[li] if 0 <= li < len(COCO_CLASSES) else "unknown"
                cx, cy, bw, bh = (float(box_tensor[j]) for j in range(4))
                dz, _ = object_distance_and_urgency(cx, cy, bw, bh, cat, _GOLD_ASSISTIVE_SPEC)
                distance[i] = dz

        scene_urgency = int(record.get("scene_urgency", meta.get("urgency", 0)))
        object_urgencies = record.get("object_urgencies")
        if isinstance(object_urgencies, list) and object_urgencies:
            for i in range(num_objs):
                scene_urgency = max(scene_urgency, int(object_urgencies[i]))

        return {
            "images": image_tensor,
            "labels": labels,
            "boxes": boxes,
            "urgency": torch.tensor(scene_urgency, dtype=torch.long),
            "distance": distance,
            "num_objects": torch.tensor(num_objs, dtype=torch.long),
            "lighting": lighting,
            "dataset_source": self.dataset_source_key or str(meta.get("dataset_id", "")),
        }
