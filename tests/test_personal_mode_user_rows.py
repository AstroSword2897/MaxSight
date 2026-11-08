"""Each user gets their own personalization row; rows are never shared between users.

User ids were reduced to ``sha256(id) % 1000``, so two users collided after 35 ids and shared the
fusion model's user embedding row. Today that row is not trained per user, but sharing it would
carry one user's adaptation into another's output the moment it is.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from app.personal_mode import PersonalMode  # noqa: E402


class _RecordingFusion:
    """Stands in for MetaFusionWeights: records the user row it receives."""

    def __init__(self, capacity: int) -> None:
        self.user_embedding = torch.nn.Embedding(capacity, 4)
        self.rows: list[int | None] = []

    def __call__(self, *, modality_embeddings, user_id, **_):
        self.rows.append(None if user_id is None else int(user_id.item()))
        return torch.zeros(1, 4), torch.ones(1, 3) / 3


def _mode(capacity: int) -> tuple[PersonalMode, _RecordingFusion]:
    mode = PersonalMode(embed_dim=4)
    fusion = _RecordingFusion(capacity)
    mode.meta_fusion = fusion
    return mode, fusion


def _fuse(mode: PersonalMode, user_id: str) -> None:
    mode.fuse_with_personalization({"vision": torch.zeros(1, 4)}, user_id=user_id)


def test_every_user_up_to_capacity_gets_a_distinct_row():
    mode, fusion = _mode(capacity=1000)
    for i in range(1000):
        _fuse(mode, f"user{i}")
    assert len(set(fusion.rows)) == 1000


def test_a_user_keeps_the_same_row():
    mode, fusion = _mode(capacity=10)
    _fuse(mode, "alice")
    _fuse(mode, "bob")
    _fuse(mode, "alice")
    assert fusion.rows[0] == fusion.rows[2] != fusion.rows[1]


def test_users_beyond_capacity_get_no_row_instead_of_someone_elses():
    mode, fusion = _mode(capacity=2)
    for name in ("a", "b", "c"):
        _fuse(mode, name)
    assert fusion.rows == [0, 1, None]


def test_no_user_means_no_row():
    mode, fusion = _mode(capacity=2)
    mode.fuse_with_personalization({"vision": torch.zeros(1, 4)}, user_id=None)
    assert fusion.rows == [None]
