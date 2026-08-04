#!/usr/bin/env python3
"""Fail CI if ml.runtime.stage_a imports networking or app-layer connectivity modules."""

from __future__ import annotations

import ast
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE_A_ROOT = REPO_ROOT / "ml" / "runtime" / "stage_a"

FORBIDDEN_MODULES = frozenset(
    {
        "boto3",
        "botocore",
        "urllib",
        "urllib.request",
        "urllib3",
        "http",
        "http.client",
        "requests",
        "aiohttp",
        "app.connectivity",
        "app.stage_b",
        "app.model_update",
        "ml.retrieval",
        "ml.pipeline",
    }
)

FORBIDDEN_PREFIXES = (
    "urllib",
    "http",
    "boto3",
    "botocore",
    "requests",
    "aiohttp",
    "app.connectivity",
    "app.stage_b",
    "app.model_update",
    "ml.retrieval",
    "ml.pipeline",
)


def _is_forbidden(name: str) -> bool:
    if name in FORBIDDEN_MODULES:
        return True
    return any(name == p or name.startswith(p + ".") for p in FORBIDDEN_PREFIXES)


def _imported_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module)
                for alias in node.names:
                    if alias.name != "*":
                        names.add(f"{node.module}.{alias.name}")
    return names


def scan_stage_a(root: Path = STAGE_A_ROOT) -> list[str]:
    """Return human-readable violations for forbidden imports under stage_a."""
    violations: list[str] = []
    if not root.is_dir():
        return [f"missing stage_a package: {root}"]
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for name in sorted(_imported_names(tree)):
            if _is_forbidden(name):
                try:
                    rel = path.relative_to(REPO_ROOT)
                except ValueError:
                    rel = path
                violations.append(
                    f"{rel}: forbidden import {name!r} "
                    "(Stage A isolation invariant — no network/connectivity)"
                )
    return violations


def run_self_test() -> int:
    """Inject a forbidden import and expect a non-zero detection."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "stage_a"
        root.mkdir()
        bad = root / "bad.py"
        bad.write_text("import requests\n", encoding="utf-8")
        violations = scan_stage_a(root)
        if not violations:
            print("self-test FAILED: expected to detect forbidden import", file=sys.stderr)
            return 1
        print("self-test OK: isolation detector caught injected violation")
        return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--self-test" in args:
        return run_self_test()
    violations = scan_stage_a()
    if violations:
        for line in violations:
            print(line, file=sys.stderr)
        return 1
    print(f"OK: Stage A isolation clean under {STAGE_A_ROOT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
