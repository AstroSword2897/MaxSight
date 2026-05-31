"""Validate IAM policies satisfy least-privilege production requirements."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

IAM_DIR = Path(__file__).resolve().parents[2] / "infra" / "iam"

REQUIRED_STUBS = (
    "sagemaker_execution_role.json",
    "s3_bucket_policy.json",
    "kms_training_volume_policy.json",
    "ssm_parameters_read_policy.json",
)

# Patterns that indicate dangerously broad permissions
_WILDCARD_ACTION_RE = re.compile(r'"Action"\s*:\s*"\*"')
_PUBLIC_PRINCIPAL_RE = re.compile(r'"Principal"\s*:\s*"\*"')


@dataclass
class PolicyViolation:
    file: str
    rule: str
    detail: str


def _check_policy(name: str, payload: dict[str, Any]) -> list[PolicyViolation]:
    violations: list[PolicyViolation] = []
    text = json.dumps(payload)
    if _WILDCARD_ACTION_RE.search(text):
        violations.append(
            PolicyViolation(name, "wildcard_action", "Action:* is not allowed in production")
        )
    statements = payload.get("Statement", [])
    if isinstance(statements, list):
        for stmt in statements:
            effect = stmt.get("Effect", "")
            principal = stmt.get("Principal", "")
            if effect == "Allow" and principal == "*":
                cond = stmt.get("Condition", {})
                if not cond:
                    violations.append(
                        PolicyViolation(
                            name, "public_principal", "Allow with Principal:* requires Condition"
                        )
                    )
    return violations


def validate_all(iam_dir: Path | None = None) -> list[PolicyViolation]:
    """Return all policy violations found under iam_dir."""
    root = iam_dir or IAM_DIR
    violations: list[PolicyViolation] = []
    for name in REQUIRED_STUBS:
        path = root / name
        if not path.exists():
            violations.append(PolicyViolation(name, "missing_file", f"{path} not found"))
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            violations.append(PolicyViolation(name, "invalid_json", str(exc)))
            continue
        violations.extend(_check_policy(name, payload))
    return violations
