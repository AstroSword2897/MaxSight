"""Tests for model-release IAM scope validation (MAXS-204c)."""

from __future__ import annotations

from scripts.infra.validate_infra_stubs import validate_model_release_iam_scope


def test_model_release_roles_are_scoped() -> None:
    errors = validate_model_release_iam_scope()
    assert errors == [], errors
