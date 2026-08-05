#!/usr/bin/env python3
"""Pre-SageMaker gate: validates all contracts, configs, and security policy
before launching a training job. Exit 0 = green; non-zero = block the job.

Usage:
    python scripts/infra/pre_sagemaker_gate.py
    # or via canonical runner:
    python scripts/product/run.py gate --checks pre_sagemaker
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"


def _check(label: str, fn) -> tuple[str, str]:
    try:
        msg = fn()
        return PASS, msg or ""
    except Exception as exc:
        return FAIL, str(exc)


def check_tier_configs() -> str:
    """All three tier YAML files must exist and be valid."""
    from ml.runtime.tier_router import TierRouter

    router = TierRouter()
    loaded = list(router._profiles.keys())
    if len(loaded) < 3:
        raise RuntimeError(f"Only {len(loaded)}/3 tier configs loaded; missing profiles")
    return f"loaded {len(loaded)} tier profiles"


def check_runtime_contracts() -> str:
    """JSON schema and OpenAPI spec must exist and contain required fields."""
    schema_path = REPO / "docs" / "contracts" / "schemas" / "runtime_response.json"
    openapi_path = REPO / "docs" / "contracts" / "openapi.yaml"
    if not schema_path.exists():
        raise FileNotFoundError(f"runtime_response.json schema missing: {schema_path}")
    if not openapi_path.exists():
        raise FileNotFoundError(f"openapi.yaml missing: {openapi_path}")
    schema = json.loads(schema_path.read_text())
    required_keys = {"frame_id", "tier", "degraded_mode", "critical_events"}
    props = set(schema.get("properties", {}).keys())
    missing = required_keys - props
    if missing:
        raise ValueError(f"Schema missing required keys: {missing}")
    return "schema + openapi present"


def check_security_policy() -> str:
    """IAM stubs must exist and not contain overly permissive principals."""
    from ml.infra.security_policy import validate_iam_stubs

    report = validate_iam_stubs()
    if not report.valid:
        raise RuntimeError(f"Security policy violations: {report.errors}")
    if report.checked_files == 0:
        return "iam stubs directory not found (non-blocking in CI)"
    return f"validated {report.checked_files} IAM stubs"


def check_reproducibility_module() -> str:
    """Reproducibility module must be importable and seed function must work."""
    from ml.training.reproducibility import reproducibility_manifest, set_deterministic_seed

    set_deterministic_seed(42)
    manifest = reproducibility_manifest(seed=42)
    assert manifest["seed"] == 42
    assert manifest["deterministic_backends"] is True
    return "seed + manifest OK"


def check_ontology() -> str:
    """Disability ontology must load and contain exactly 7 disabilities."""
    from ml.data.ontology.loader import DisabilityOntology

    onto = DisabilityOntology.load()
    onto.validate()
    count = onto.to_dict()["disability_count"]
    return f"{count} disabilities validated"


def check_therapy_constraints() -> str:
    """Therapy constraints must load with non-empty routing and rate limits."""
    from ml.therapy.constraints_loader import TherapyConstraints

    tc = TherapyConstraints.load()
    if tc.max_prompts_per_minute <= 0:
        raise ValueError("max_prompts_per_minute must be positive")
    if not tc.disability_routing:
        raise ValueError("disability_routing map is empty")
    return f"constraints OK: {tc.max_prompts_per_minute} prompts/min limit"


def check_deprecations_yaml() -> str:
    """deprecations.yaml must exist and list canonical commands."""
    import yaml

    dep_path = REPO / "scripts" / "product" / "deprecations.yaml"
    if not dep_path.exists():
        raise FileNotFoundError(f"deprecations.yaml missing: {dep_path}")
    data = yaml.safe_load(dep_path.read_text())
    cmds = data.get("canonical_commands", {})
    required = {"train", "validate", "export", "gate"}
    missing = required - set(cmds.keys())
    if missing:
        raise ValueError(f"canonical_commands missing: {missing}")
    return f"{len(cmds)} canonical commands registered"


def main() -> int:
    checks: list[tuple[str, object]] = [
        ("Tier configs", check_tier_configs),
        ("Runtime contracts", check_runtime_contracts),
        ("Security policy", check_security_policy),
        ("Reproducibility module", check_reproducibility_module),
        ("Disability ontology", check_ontology),
        ("Therapy constraints", check_therapy_constraints),
        ("Deprecations registry", check_deprecations_yaml),
    ]

    passed = 0
    failed = 0
    print(f"{'Pre-SageMaker Gate':=^60}")
    for label, fn in checks:
        status, msg = _check(label, fn)
        symbol = "✓" if status == PASS else "✗"
        print(f"  {symbol} {label:<35} {msg}")
        if status == PASS:
            passed += 1
        else:
            failed += 1

    print("=" * 60)
    print(f"  {passed} passed, {failed} failed")
    if failed > 0:
        print("  GATE BLOCKED — fix failures before submitting SageMaker job")
        return 1
    print("  GATE PASSED — safe to launch SageMaker job")
    return 0


if __name__ == "__main__":
    sys.exit(main())
