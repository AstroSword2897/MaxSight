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


def main() -> int:
    strict = os.environ.get("MAXSIGHT_INFRA_STRICT_PLACEHOLDERS", "").strip() == "1"
    errors = validate_parse_only()
    errors.extend(validate_sagemaker_role_shape())
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
