"""Semantic relations must use the same embedding row in every process.

Python's built-in ``hash()`` of a string is salted per process (PYTHONHASHSEED), so a model trained in
one process looked up different relation rows at inference. The classifier's own prediction
``semantic_k`` must use row ``num_spatial + k``; rule predicates must map the same way everywhere.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]

_PROBE = """
import json, torch
from ml.models.scene_graph.scene_graph_encoder import SceneGraphEncoder
torch.manual_seed(0)
enc = SceneGraphEncoder(object_embed_dim=8, relation_embed_dim=4, semantic_rules={("cup", "table"): "on_top_of"})
enc.eval()
boxes = torch.tensor([[0.1, 0.1, 0.2, 0.2], [0.5, 0.5, 0.9, 0.9], [0.3, 0.6, 0.4, 0.7]])
emb = torch.randn(3, 8)
with torch.no_grad():
    rels, _, attr = enc.extract_relations(boxes, emb, ["cup", "table", "door"])
rows = [int((enc.relation_embedding.weight == attr[e]).all(dim=1).nonzero()[0]) for e in range(attr.shape[0])]
print(json.dumps([[r.predicate, row] for r, row in zip(rels, rows)]))
"""


def _rows(hash_seed: str) -> list[list]:
    env = {**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONPATH": str(REPO)}
    out = subprocess.run([sys.executable, "-c", _PROBE], cwd=REPO, env=env, capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_relation_rows_do_not_depend_on_the_process_hash_seed():
    assert _rows("1") == _rows("2") == _rows("12345")


def test_classifier_prediction_semantic_k_uses_row_num_spatial_plus_k():
    for predicate, row in _rows("7"):
        if predicate.startswith("semantic_"):
            assert row == 6 + int(predicate.split("_")[1]), (predicate, row)


def test_rule_predicates_stay_inside_the_semantic_rows():
    rule_rows = [row for predicate, row in _rows("3") if predicate == "on_top_of"]
    assert rule_rows and all(6 <= row < 16 for row in rule_rows)
