#!/usr/bin/env python3
"""Run MaxSight Stress Test Suite

Executes comprehensive stress tests to validate system stability and safety."""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from ml.training.stress_tests import StressTestSuite, StressTestConfig
from ml.utils.error_handling import HeadKillSwitchManager, EthicalGuard
from ml.utils.monitoring import HealthChecker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Run MaxSight stress tests')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='stress_test_report.json',
                       help='Output path for stress test report')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick tests only (skip expensive tests)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run tests on')
    
    args = parser.parse_args()
    
    logger.info("Starting MaxSight Stress Test Suite")
    logger.info(f"Device: {args.device}")
    logger.info(f"Quick mode: {args.quick}")
    
    # Load model
    if args.checkpoint:
        logger.info(f"Loading model from {args.checkpoint}")
        # FUTURE ENHANCEMENT: Load actual trained model for stress testing.
        model = None  # Placeholder
    else:
        logger.warning("No checkpoint provided, using dummy model")
        model = None  # Placeholder
    
    if model is None:
        logger.error("Model loading not implemented. Please provide a valid model.")
        return
    
    # Create stress test config
    config = StressTestConfig()
    if args.quick:
        # Reduce test scope for quick mode
        config.head_isolation_variants = [['detection'], ['all']]  # Only A and E
        config.loss_scaling_factors = [1.0, 5.0]  # Only test extremes
        config.corruption_types = ['gaussian_blur', 'random_occlusion']  # Fewer corruptions
    
    # Create kill switch manager
    kill_switch_manager = HeadKillSwitchManager()
    
    # Create stress test suite
    suite = StressTestSuite(config)
    
    # Create dummy data loaders (replace with actual loaders)
    train_loader = None
    val_loader = None
    
    # Run health check first (Tier 1 heads)
    logger.info("Running health check (Tier 1 heads)...")
    health_checker = HealthChecker(model, device=args.device)
    health_report = health_checker.run_full_check()
    
    if health_report['overall_status'] == 'FAIL':
        logger.error("❌ Health check FAILED - Tier 1 heads not safe")
        logger.error("Failures:")
        for check_name, check_results in health_report['checks'].items():
            if 'failures' in check_results:
                for failure in check_results['failures']:
                    logger.error(f"  - {check_name}: {failure}")
        logger.error("⚠️  Do not proceed with stress tests until Tier 1 is fixed")
        sys.exit(1)
    else:
        logger.info("✅ Health check passed - Tier 1 heads safe")
    
    # Run stress tests
    logger.info("Running stress test suite...")
    results = suite.run_all(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        kill_switch_manager=kill_switch_manager
    )
    
    # Generate and save report
    logger.info("Generating stress test report...")
    suite.save_report(args.output)
    
    # Print dashboard summary
    dashboard = results['dashboard']
    summary = dashboard['summary']
    
    print("\n" + "="*60)
    print("STRESS TEST SUMMARY")
    print("="*60)
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed']} ✅")
    print(f"Failed: {summary['failed']} ❌")
    print(f"Warnings: {summary['warnings']} ⚠️")
    print("="*60)
    
    # Print failed tests
    if summary['failed'] > 0:
        print("\nFAILED TESTS:")
        for test in dashboard['tests']:
            if test['status'] == '❌':
                print(f"  - {test['category']}/{test['test']}")
                if test['red_flags']:
                    for flag in test['red_flags']:
                        print(f"    🚩 {flag}")
    
    # Print warnings
    if summary['warnings'] > 0:
        print("\nWARNINGS:")
        for test in dashboard['tests']:
            if test['red_flags']:
                print(f"  - {test['test']}:")
                for flag in test['red_flags']:
                    print(f"    ⚠️ {flag}")
    
    print(f"\nFull report saved to: {args.output}")
    
    # Exit with error code if tests failed
    if summary['failed'] > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()

