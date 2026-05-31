"""Gold training plane: canonical JSONL manifests + lazy tensor materialization."""

from ml.data.gold.builder import (
    build_gold_jsonl_from_adapter,
    build_gold_manifest,
    finalize_gold_record,
    validate_gold_line,
    validate_gold_line_in_memory,
    write_manifest_meta,
)
from ml.data.gold.dataNormalizationLayer import (
    COCOAdapter,
    MaxSightListAdapter,
    VideoManifestAdapter,
)
from ml.data.gold.dataset import GoldManifestDataset, load_gold_meta
from ml.data.gold.errors import GoldConfigError
from ml.data.gold.io import GoldIOError, ShardReader, verify_shard_sha256
from ml.data.gold.label_mapper import LabelMapper
from ml.data.gold.schema import (
    GOLD_LINE_SCHEMA_VERSION,
    GOLD_META_SCHEMA_VERSION,
    LABEL_SPACE_ACCESSIBILITY_622,
    validate_meta,
)

__all__ = [
    "GOLD_LINE_SCHEMA_VERSION",
    "GOLD_META_SCHEMA_VERSION",
    "LABEL_SPACE_ACCESSIBILITY_622",
    "GoldConfigError",
    "GoldIOError",
    "ShardReader",
    "LabelMapper",
    "MaxSightListAdapter",
    "COCOAdapter",
    "VideoManifestAdapter",
    "finalize_gold_record",
    "build_gold_manifest",
    "build_gold_jsonl_from_adapter",
    "write_manifest_meta",
    "validate_gold_line",
    "validate_gold_line_in_memory",
    "validate_meta",
    "GoldManifestDataset",
    "load_gold_meta",
    "verify_shard_sha256",
]
