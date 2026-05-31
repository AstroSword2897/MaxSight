"""Contract tests for ml/training/configs/registry/label_spaces.yaml."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.data.label_space_registry import (  # noqa: E402
    LABEL_SPACE_REGISTRY_SCHEMA_VERSION,
    default_label_space_registry_path,
    load_label_space_registry,
)


def test_label_space_registry_loads() -> None:
    reg = load_label_space_registry(default_label_space_registry_path(PROJECT_ROOT))
    assert reg.schema_version == LABEL_SPACE_REGISTRY_SCHEMA_VERSION
    coco = reg.resolve("coco_80")
    assert coco.num_classes == 80
    acc = reg.resolve("accessibility_622")
    assert acc.num_classes == 622
    assert acc.parent == "coco_80"
