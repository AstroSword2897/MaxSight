"""Regression guards for PR #14 CI failures.

Every test here must stay torch-free: it runs in the slim ``contracts`` CI profile.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

# Modules that earlier failed in CI because a package __init__ imported torch eagerly.
SLIM_MODULES = [
    "app",
    "app.connectivity.monitor",
    "app.stage_b.client",
    "app.model_update.staging",
    "app.model_update.activation",
    "app.ui.hazard_haptics",
    "ml.data",
    "ml.data.dataset_registry",
    "ml.data.ontology.loader",
    "ml.pipeline",
    "ml.pipeline.sagemaker_config",
    "ml.runtime_constants",
    "ml.infra.security_policy",
]


def _run_without_torch(code: str) -> subprocess.CompletedProcess[str]:
    """Run ``code`` in a fresh interpreter where importing torch raises ModuleNotFoundError."""
    prelude = textwrap.dedent(
        """
        import importlib.abc, sys
        class _NoTorch(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name == "torch" or name.startswith("torch."):
                    raise ModuleNotFoundError("No module named 'torch' (blocked by regression test)")
        sys.meta_path.insert(0, _NoTorch())
        """
    )
    return subprocess.run(
        [sys.executable, "-c", prelude + textwrap.dedent(code)],
        cwd=REPO, capture_output=True, text=True, timeout=120,
    )


@pytest.mark.parametrize("module", SLIM_MODULES)
def test_module_imports_without_torch(module):
    proc = _run_without_torch(f"import importlib; importlib.import_module({module!r})")
    assert proc.returncode == 0, proc.stderr[-800:]


@pytest.mark.parametrize("package", ["app", "ml.data", "ml.pipeline"])
def test_lazy_package_rejects_unknown_attribute(package):
    mod = importlib.import_module(package)
    with pytest.raises(AttributeError):
        mod.definitely_not_an_export  # noqa: B018


@pytest.mark.parametrize("package", ["ml.data", "ml.pipeline"])
def test_lazy_exports_point_at_real_modules_and_match_all(package):
    mod = importlib.import_module(package)
    exports: dict[str, str] = mod._LAZY_EXPORTS
    assert set(mod.__all__) == set(exports)
    for name, target in exports.items():
        assert importlib.util.find_spec(target) is not None, f"{package}.{name} -> missing module {target}"


def test_importing_light_package_does_not_load_torch_eagerly():
    proc = _run_without_torch(
        """
        import sys
        import app, ml.data, ml.pipeline
        assert "torch" not in sys.modules
        """
    )
    assert proc.returncode == 0, proc.stderr[-800:]


# ── IAM stubs / ontology (pre-SageMaker gate) ────────────────────────────────


def test_required_iam_stubs_exist_and_pass_policy_check():
    from ml.infra.security_policy import REQUIRED_IAM_FILES, validate_iam_stubs

    for name in REQUIRED_IAM_FILES:
        assert (REPO / "infra" / "iam" / name).is_file(), name
    report = validate_iam_stubs()
    assert report.valid, report.errors
    assert report.checked_files == len(REQUIRED_IAM_FILES)


def test_iam_stubs_contain_no_wildcard_actions():
    for path in (REPO / "infra" / "iam").glob("*.json"):
        text = path.read_text()
        assert '"Action": "*"' not in text, path.name
        assert '"s3:*"' not in text or "Deny" in text, f"{path.name}: s3:* only allowed in a Deny statement"


def test_ontology_is_complete_and_uses_known_condition_keys():
    from ml.data.ontology.loader import DisabilityOntology
    from ml.runtime_constants import CONDITION_MODE_IDS

    onto = DisabilityOntology.load()
    onto.validate()
    ids = onto.to_dict()["ids"]
    assert len(ids) == 7
    for did in ids:
        entry = onto.get(did)
        assert entry.therapy_focus, f"{did}: empty therapy_focus"
        assert entry.model_condition_key in CONDITION_MODE_IDS, entry.model_condition_key


def test_pre_sagemaker_gate_passes_without_torch():
    proc = subprocess.run(
        [sys.executable, "scripts/infra/pre_sagemaker_gate.py"],
        cwd=REPO, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stdout[-1200:] + proc.stderr[-400:]


# ── Version-control hygiene: required JSON must not be gitignored again ──────

TRACKED_JSON = [
    "infra/iam/sagemaker_execution_role.json",
    "infra/iam/s3_bucket_policy.json",
    "infra/iam/kms_training_volume_policy.json",
    "ml/data/ontology/disability_ontology.json",
]


@pytest.mark.parametrize("rel", TRACKED_JSON)
def test_required_json_is_not_gitignored(rel):
    if not (REPO / ".git").exists():
        pytest.skip("not a git checkout")
    proc = subprocess.run(["git", "check-ignore", "-q", rel], cwd=REPO)
    assert proc.returncode == 1, f"{rel} is matched by .gitignore (the *.json rule), so CI would not see it"


# ── Pyright / pytest config ──────────────────────────────────────────────────


def test_pyright_config_has_no_machine_specific_paths():
    cfg = json.loads((REPO / "pyrightconfig.json").read_text())
    assert "venvPath" not in cfg and "venv" not in cfg
    for env in cfg.get("executionEnvironments", []):
        assert not env.get("extraPaths"), env
    text = (REPO / "pyproject.toml").read_text()
    pyright_section = text.split("[tool.pyright]", 1)[1].split("\n[", 1)[0]
    for forbidden in ("venvPath", "extraPaths", "../2026"):
        assert forbidden not in pyright_section, forbidden


def test_benchmark_marker_registered_and_latency_tests_tagged():
    pyproject = (REPO / "pyproject.toml").read_text()
    assert "benchmark:" in pyproject
    for rel in ("tests/test_performance.py", "tests/test_multihead_benchmark.py"):
        assert "@pytest.mark.benchmark" in (REPO / rel).read_text(), rel
