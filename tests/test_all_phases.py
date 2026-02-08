"""Master Test Runner for All Phases (0-5) Runs comprehensive tests for all Phase 0-5 components."""

import pytest
import sys
from pathlib import Path

# Add parent directory to path.
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_all_phase_tests():
    """Run all Phase 0-5 tests."""
    print("=" * 80)
    print("MaxSight 3.0: Comprehensive Test Suite for Phases 0-5")
    print("=" * 80)
    print()
    
    test_files = [
        "tests/test_phase0_backbone.py",
        "tests/test_phase1_fusion.py",
        "tests/test_phase2_heads.py",
        "tests/test_phase3_retrieval.py",
        "tests/test_phase4_knowledge.py",
        "tests/test_phase5_training.py",
    ]
    
    results = []
    
    for test_file in test_files:
        print(f"\n{'=' * 80}")
        print(f"Running: {test_file}")
        print('=' * 80)
        
        try:
            exit_code = pytest.main([test_file, "-v", "--tb=short"])
            results.append((test_file, exit_code == 0))
        except Exception as e:
            print(f"Error running {test_file}: {e}")
            results.append((test_file, False))
    
    # Summary.
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_file, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{status}: {test_file}")
    
    print(f"\nTotal: {passed}/{total} test suites passed")
    
    if passed == total:
        print("All test suites passed!")
        return 0
    else:
        print("WARNING Some test suites failed - check output above")
        return 1


if __name__ == "__main__":
    exit(run_all_phase_tests())







