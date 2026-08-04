"""Condition tensor contract tests (MAXS-302a)."""

from __future__ import annotations

import json
from pathlib import Path

from ml.runtime_constants import (
    CONDITION_MODE_IDS,
    CONDITION_TENSOR_WIDTH,
    condition_mode_to_tensor_index,
)

SCHEMA = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "contracts"
    / "schemas"
    / "condition_tensor.json"
)


def test_condition_ids_cover_width() -> None:
    assert CONDITION_TENSOR_WIDTH == len(CONDITION_MODE_IDS)
    assert condition_mode_to_tensor_index("glaucoma") == CONDITION_MODE_IDS["glaucoma"]
    assert condition_mode_to_tensor_index(None) == 0


def test_condition_schema_exists() -> None:
    data = json.loads(SCHEMA.read_text(encoding="utf-8"))
    assert data["properties"]["encoding"]["const"] == "one_hot"
    assert data["properties"]["tensor_name"]["const"] == "condition_tensor"
