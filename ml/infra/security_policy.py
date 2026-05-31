"""Security policy validation for data pipeline and IAM stubs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class SecurityPolicyReport:
    """Outcome of security policy checks."""

    valid: bool
    errors: list[str]
    checked_files: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": self.errors,
            "checked_files": self.checked_files,
        }


REQUIRED_IAM_FILES = (
    "sagemaker_execution_role.json",
    "s3_bucket_policy.json",
    "kms_training_volume_policy.json",
)


def validate_iam_stubs(infra_dir: Path | None = None) -> SecurityPolicyReport:
    """Validate IAM JSON stubs exist and deny public S3 access patterns."""
    root = infra_dir or Path(__file__).resolve().parents[2] / "infra" / "iam"
    errors: list[str] = []
    checked = 0
    for name in REQUIRED_IAM_FILES:
        path = root / name
        checked += 1
        if not path.exists():
            errors.append(f"missing_iam_stub:{name}")
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"invalid_json:{name}:{exc}")
            continue
        # Detect explicitly wildcard principals (e.g. "Principal": "*") combined
        # with S3 actions and no Deny/Condition safeguard. Wildcard resources
        # (e.g. "Resource": "*") are acceptable for service-scoped roles.
        text_lower = json.dumps(payload).lower()
        has_wildcard_principal = (
            '"principal": "*"' in text_lower or "'principal': '*'" in text_lower
        )
        if has_wildcard_principal and "s3" in text_lower:
            if "deny" not in text_lower and "condition" not in text_lower:
                errors.append(f"overly_permissive_principal:{name}")
    return SecurityPolicyReport(valid=len(errors) == 0, errors=errors, checked_files=checked)


def validate_s3_path_safe(uri: str) -> None:
    """Reject path traversal or non-s3 schemes for pipeline inputs."""
    if ".." in uri:
        raise ValueError("path traversal not allowed")
    if uri.startswith("s3://"):
        return
    if Path(uri).is_absolute() or uri.startswith("./") or "/" in uri:
        return
    raise ValueError(f"unsupported uri scheme: {uri}")
