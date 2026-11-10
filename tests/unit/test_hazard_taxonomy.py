"""The V1 hazard taxonomy (configs/hazard/taxonomy.yaml) keeps class, category and urgency separate.

These checks implement the "done when" rules of ticket 301:
  * class, category and urgency are three separate fields;
  * every class has a critical flag;
  * every detector label is mapped to exactly one hazard class, or listed as unmapped with a reason;
  * the Data Role Rule is written down;
  * the file stays "proposed" until it is approved and versioned 1.0.0.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
TAXONOMY = yaml.safe_load((REPO / "configs" / "hazard" / "taxonomy.yaml").read_text(encoding="utf-8"))
LABEL_MAP = yaml.safe_load((REPO / "configs" / "data" / "label_map.yaml").read_text(encoding="utf-8"))

CATEGORIES = {"dynamic", "static", "environmental", "context_dependent"}
CLASSES = TAXONOMY["classes"]


def test_categories_are_the_four_confirmed_by_dr_018():
    assert set(TAXONOMY["categories"]) == CATEGORIES


def test_urgency_is_a_separate_zero_to_three_scale():
    assert TAXONOMY["urgency"]["levels"] == {0: "informational", 1: "caution", 2: "high", 3: "critical"}


@pytest.mark.parametrize("class_name", sorted(CLASSES))
def test_a_class_carries_no_urgency(class_name):
    """Urgency comes from the rule table at run time; a class must never fix it."""
    assert not any("urgency" in key for key in CLASSES[class_name])


@pytest.mark.parametrize("class_name", sorted(CLASSES))
def test_every_class_has_a_critical_flag(class_name):
    assert isinstance(CLASSES[class_name].get("critical"), bool)


@pytest.mark.parametrize("class_name", sorted(CLASSES))
def test_a_class_only_lists_known_categories(class_name):
    allowed = CLASSES[class_name]["allowed_categories"]
    assert allowed, f"{class_name} lists no categories"
    assert set(allowed) <= CATEGORIES


def test_class_names_and_category_names_do_not_overlap():
    assert not set(CLASSES) & CATEGORIES


@pytest.mark.parametrize("class_name", sorted(CLASSES))
def test_every_class_has_include_and_exclude_examples(class_name):
    assert CLASSES[class_name]["include"], f"{class_name}: no include examples"
    assert CLASSES[class_name]["exclude"], f"{class_name}: no exclude examples"


def test_every_detector_label_is_mapped_once_or_explicitly_unmapped():
    mapped: dict[str, str] = {}
    for class_name, spec in CLASSES.items():
        for label in spec["detector_labels"]:
            assert label not in mapped, f"{label} is mapped to both {mapped[label]} and {class_name}"
            mapped[label] = class_name

    unmapped = TAXONOMY.get("unmapped_detector_labels") or {}
    detector_labels = set(LABEL_MAP["classes"])

    assert set(mapped) <= detector_labels, f"unknown detector labels: {sorted(set(mapped) - detector_labels)}"
    missing = detector_labels - set(mapped) - set(unmapped)
    assert not missing, f"detector labels with no hazard class and no reason: {sorted(missing)}"
    assert all(reason for reason in unmapped.values())


def test_the_data_role_rule_keeps_safety_evaluation_out_of_training():
    roles = TAXONOMY["data_roles"]
    assert set(roles) == {"training", "validation", "independent_safety_evaluation"}
    safety_use = roles["independent_safety_evaluation"]["allowed_use"].lower()
    assert "sg-01" in safety_use and "sg-02" in safety_use
    assert "never used for training" in safety_use


def test_the_taxonomy_stays_proposed_until_approved_as_1_0_0():
    """Approval is an owner decision: the status and the version must change together."""
    if TAXONOMY["status"] == "approved":
        assert TAXONOMY["schema_version"] == "1.0.0"
    else:
        assert TAXONOMY["status"] == "proposed"
        assert TAXONOMY["schema_version"].startswith("0.")
