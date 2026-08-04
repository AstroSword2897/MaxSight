#!/usr/bin/env python3
"""Validate infra/ JSON files parse; optionally fail on unreplaced {{placeholders}}."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INFRA_ROOT = REPO_ROOT / "infra"


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_json_files() -> list[Path]:
    return sorted(p for p in INFRA_ROOT.rglob("*.json") if p.is_file())


def validate_parse_only() -> list[str]:
    errors: list[str] = []
    for path in _collect_json_files():
        try:
            _load_json(path)
        except json.JSONDecodeError as e:
            errors.append(f"{path.relative_to(REPO_ROOT)}: {e}")
    return errors


ALLOW_PLACEHOLDER_FILES = frozenset({"infra/s3/bucket_encryption_sse_kms.json"})


def validate_no_placeholders() -> list[str]:
    errors: list[str] = []
    for path in _collect_json_files():
        rel = str(path.relative_to(REPO_ROOT))
        if rel in ALLOW_PLACEHOLDER_FILES:
            continue
        text = path.read_text(encoding="utf-8")
        if "{{" in text or "}}" in text:
            errors.append(f"{rel}: contains {{placeholders}} — replace before apply.")
    return errors


def validate_sagemaker_role_shape() -> list[str]:
    path = INFRA_ROOT / "iam" / "sagemaker_execution_role.json"
    if not path.exists():
        return []
    data = _load_json(path)
    if not isinstance(data, dict):
        return [f"{path}: root must be an object."]
    missing = [k for k in ("trust_policy", "inline_permissions") if k not in data]
    if missing:
        return [f"{path}: missing keys {missing}."]
    for key in ("trust_policy", "inline_permissions"):
        stmt = data[key]
        if not isinstance(stmt, dict):
            return [f"{path}: {key} must be an object."]
        if stmt.get("Version") != "2012-10-17":
            return [f"{path}: {key}.Version should be 2012-10-17."]
    return []


MODEL_RELEASE_ROLE_FILES = (
    "infra/iam/model_release_export_role.json",
    "infra/iam/model_release_phone_readonly_role.json",
)


def _statement_resources(stmt: object) -> list[str]:
    if not isinstance(stmt, dict):
        return []
    resource = stmt.get("Resource", [])
    if isinstance(resource, str):
        return [resource]
    if isinstance(resource, list):
        return [r for r in resource if isinstance(r, str)]
    return []


def validate_model_release_iam_scope() -> list[str]:
    """Fail if model-release roles allow S3 outside model-release/* (except Deny)."""
    errors: list[str] = []
    for rel in MODEL_RELEASE_ROLE_FILES:
        path = REPO_ROOT / rel
        if not path.exists():
            errors.append(f"{rel}: missing model-release IAM stub")
            continue
        data = _load_json(path)
        if not isinstance(data, dict):
            errors.append(f"{rel}: root must be an object")
            continue
        perms = data.get("inline_permissions")
        if not isinstance(perms, dict):
            errors.append(f"{rel}: missing inline_permissions")
            continue
        statements = perms.get("Statement", [])
        if not isinstance(statements, list):
            errors.append(f"{rel}: Statement must be a list")
            continue
        for stmt in statements:
            if not isinstance(stmt, dict):
                continue
            if stmt.get("Effect") == "Deny":
                continue
            for resource in _statement_resources(stmt):
                if resource.endswith(":{{RELEASE_BUCKET}}") or resource.endswith(
                    ":{{RELEASE_BUCKET}}/"
                ):
                    # ListBucket on bucket ARN is allowed when Condition scopes prefix.
                    continue
                if "/model-release/" not in resource and not resource.endswith(
                    "/model-release/*"
                ):
                    if "s3:::" in resource and "{{RELEASE_BUCKET}}" in resource:
                        # Bucket-level ListBucket without object path is OK.
                        if resource.rstrip("/").endswith("{{RELEASE_BUCKET}}"):
                            continue
                    errors.append(
                        f"{rel}: Allow resource widens beyond model-release/*: {resource}"
                    )
        # Phone role must not allow PutObject.
        if rel.endswith("phone_readonly_role.json"):
            for stmt in statements:
                if not isinstance(stmt, dict) or stmt.get("Effect") != "Allow":
                    continue
                actions = stmt.get("Action", [])
                if isinstance(actions, str):
                    actions = [actions]
                if any(a in {"s3:PutObject", "s3:*"} for a in actions):
                    errors.append(f"{rel}: phone role must not Allow PutObject/s3:*")
    return errors


def main() -> int:
    strict = os.environ.get("MAXSIGHT_INFRA_STRICT_PLACEHOLDERS", "").strip() == "1"
    errors = validate_parse_only()
    errors.extend(validate_sagemaker_role_shape())
    errors.extend(validate_model_release_iam_scope())
    if strict:
        errors.extend(validate_no_placeholders())
    if errors:
        for line in errors:
            print(line, file=sys.stderr)
        return 1
    print(f"OK: {len(_collect_json_files())} JSON file(s) under infra/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
