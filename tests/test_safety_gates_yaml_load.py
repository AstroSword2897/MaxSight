"""Load safety_gates.yaml (MAXS-301a)."""

from __future__ import annotations

from pathlib import Path

import yaml

CFG = Path(__file__).resolve().parents[1] / "ml" / "config" / "safety_gates.yaml"


def test_safety_gates_yaml_loads() -> None:
    data = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert data["thresholds"]["hazard_recall_min"] == 0.95
    assert "SG-01" in data["require_hazard_ground_truth_for"]
