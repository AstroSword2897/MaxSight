"""checkpoint_content_hash identifies a state dict by its keys, dtypes, shapes and values.

Hashing raw bytes alone lets two different checkpoints share a hash (same bytes, other shape or
dtype; key/value boundaries shifted) and crashed on bfloat16, which numpy cannot represent.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ml.training.reproducibility import CHECKPOINT_HASH_SCHEME, checkpoint_content_hash  # noqa: E402


def test_same_bytes_different_shape_hash_differently():
    values = torch.arange(6, dtype=torch.float32)
    assert checkpoint_content_hash({"w": values.reshape(2, 3)}) != checkpoint_content_hash(
        {"w": values.reshape(3, 2)}
    )


def test_same_bytes_different_dtype_hash_differently():
    raw = torch.arange(4, dtype=torch.int32)
    assert checkpoint_content_hash({"w": raw}) != checkpoint_content_hash({"w": raw.view(torch.float32)})


def test_key_value_boundary_cannot_be_shifted():
    # Without framing both hash the byte stream b"ab": key "a" + value b"b" (98) vs key "ab" + no bytes.
    a = {"a": torch.tensor([98], dtype=torch.uint8)}
    b = {"ab": torch.tensor([], dtype=torch.uint8)}
    assert checkpoint_content_hash(a) != checkpoint_content_hash(b)


def test_bfloat16_state_dict_hashes():
    digest = checkpoint_content_hash({"w": torch.ones(2, 2, dtype=torch.bfloat16)})
    assert len(digest) == 64


def test_non_contiguous_view_hashes_like_its_values():
    base = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    assert checkpoint_content_hash({"w": base.t()}) == checkpoint_content_hash({"w": base.t().contiguous()})


def test_key_order_does_not_matter_and_hash_is_stable():
    x, y = torch.ones(2), torch.zeros(3)
    assert checkpoint_content_hash({"x": x, "y": y}) == checkpoint_content_hash({"y": y, "x": x})


def test_value_change_changes_hash():
    assert checkpoint_content_hash({"w": torch.zeros(3)}) != checkpoint_content_hash(
        {"w": torch.tensor([0.0, 0.0, 1e-30])}
    )


def test_scheme_is_named_for_manifests():
    assert CHECKPOINT_HASH_SCHEME.startswith("sha256")
