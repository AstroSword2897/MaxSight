"""Structural Tests for Integration Features."""

from pathlib import Path


def test_gradnorm_import_structure():
    """Test that GradNorm import structure is correct."""
    train_loop_path = Path(__file__).parent.parent / "ml" / "training" / "train_loop.py"
    content = train_loop_path.read_text()
    assert "from ml.training.task_balancing import GradNormMultiHeadLoss" in content, (
        "GradNorm import statement missing"
    )
    assert "GRADNORM_AVAILABLE" in content, "GRADNORM_AVAILABLE availability check missing"


def test_gradnorm_parameters():
    """Test that GradNorm parameters are in __init__."""
    train_loop_path = Path(__file__).parent.parent / "ml" / "training" / "train_loop.py"
    content = train_loop_path.read_text()
    assert "use_gradnorm" in content, "use_gradnorm parameter missing"
    assert "gradnorm_alpha" in content, "gradnorm_alpha parameter missing"
    assert "gradnorm_update_interval" in content, "gradnorm_update_interval parameter missing"


def test_gradnorm_integration_code():
    """Test that GradNorm integration code exists."""
    train_loop_path = Path(__file__).parent.parent / "ml" / "training" / "train_loop.py"
    content = train_loop_path.read_text()
    assert "self.use_gradnorm" in content, "GradNorm initialization code missing"
    assert "gradnorm_loss" in content or "GradNorm" in content, "GradNorm loss computation missing"


def test_timing_import():
    """Test that time module is imported in maxsight_cnn."""
    model_path = Path(__file__).parent.parent / "ml" / "models" / "maxsight_cnn.py"
    content = model_path.read_text()
    assert "import time" in content, "time module import missing from maxsight_cnn"


def test_timing_code():
    """Test that timing enforcement code exists."""
    model_path = Path(__file__).parent.parent / "ml" / "models" / "maxsight_cnn.py"
    content = model_path.read_text()
    assert "stage_a_start_time" in content, "stage_a timing tracker missing"
    assert "stage_a_latency_ms" in content, "stage_a latency measurement missing"
    assert "skip_stage_b" in content and "latency" in content.lower(), (
        "stage_b early-exit latency gate missing"
    )
    assert "outputs[" in content, "timing metrics not written to outputs dict"


def test_timing_threshold():
    """Test that a latency threshold gate exists in maxsight_cnn."""
    model_path = Path(__file__).parent.parent / "ml" / "models" / "maxsight_cnn.py"
    content = model_path.read_text()
    assert "stage_a_latency_ms" in content, "stage_a_latency_ms variable not found"
    assert "max_latency" in content, "max_latency threshold variable not found"
    assert "skip_stage_b" in content or "skipping Stage B" in content, "Stage B skip logic missing"


def test_output_structure():
    """Test that output structure includes timing metrics."""
    model_path = Path(__file__).parent.parent / "ml" / "models" / "maxsight_cnn.py"
    content = model_path.read_text()
    assert "'stage_a_completed'" in content or '"stage_a_completed"' in content, (
        "stage_a_completed output key missing"
    )
    assert "'stage_b_completed'" in content or '"stage_b_completed"' in content, (
        "stage_b_completed output key missing"
    )
    assert "'skip_stage_b_reason'" in content or '"skip_stage_b_reason"' in content, (
        "skip_stage_b_reason output key missing"
    )
    assert "'stage_a_latency_ms'" in content or '"stage_a_latency_ms"' in content, (
        "stage_a_latency_ms output key missing"
    )


_ALL_TESTS = [
    ("GradNorm Import", test_gradnorm_import_structure),
    ("GradNorm Parameters", test_gradnorm_parameters),
    ("GradNorm Integration", test_gradnorm_integration_code),
    ("Timing Import", test_timing_import),
    ("Timing Code", test_timing_code),
    ("Timing Threshold", test_timing_threshold),
    ("Output Structure", test_output_structure),
]


def main():
    """Run all structural tests from the command line."""
    print("=" * 60)
    print("Integration Structure Tests")
    print("(Tests code structure without requiring PyTorch)")
    print("=" * 60)
    passed = 0
    for name, fn in _ALL_TESTS:
        try:
            fn()
            print(f"PASS: {name}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {name} — {e}")
    print(f"\nTotal: {passed}/{len(_ALL_TESTS)} tests passed")
    return 0 if passed == len(_ALL_TESTS) else 1


if __name__ == "__main__":
    exit(main())
