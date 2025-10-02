#!/usr/bin/env python3
"""
Comprehensive test suite for Sprint 7 - Production Enhancements

This script runs all Sprint 7 tests and generates a combined report.
"""

import sys
import os
import time
import subprocess
import tempfile
from pathlib import Path

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

def run_test(test_file, test_name):
    """Run a single test and capture results"""
    print(f"\n{'='*60}")
    print(f"Running {test_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    success = True
    error_msg = ""
    
    try:
        # Run the test
        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            print(f"✓ {test_name} completed successfully")
            print(f"  Duration: {time.time() - start_time:.2f} seconds")
        else:
            success = False
            error_msg = result.stderr or result.stdout
            print(f"✗ {test_name} failed")
            print(f"  Error: {error_msg[:500]}...")
            
    except subprocess.TimeoutExpired:
        success = False
        error_msg = "Test timed out after 5 minutes"
        print(f"✗ {test_name} timed out")
        
    except Exception as e:
        success = False
        error_msg = str(e)
        print(f"✗ {test_name} failed with exception: {e}")
    
    return {
        'name': test_name,
        'success': success,
        'duration': time.time() - start_time,
        'error': error_msg
    }

def main():
    """Main test runner"""
    print("FUA Sprint 7 - Production Enhancements Test Suite")
    print("=" * 60)
    
    # Define tests to run
    tests = [
        ("test_sprint7_monitoring.py", "Model Monitoring System"),
        ("test_sprint7_ab_testing.py", "A/B Testing Framework"),
        ("test_sprint7_degradation.py", "Performance Degradation Detection"),
        ("test_sprint7_rollback.py", "Auto Rollback System")
    ]
    
    # Run all tests
    results = []
    for test_file, test_name in tests:
        result = run_test(test_file, test_name)
        results.append(result)
    
    # Generate summary report
    print(f"\n{'='*60}")
    print("Test Suite Summary")
    print(f"{'='*60}")
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r['success'])
    failed_tests = total_tests - passed_tests
    
    print(f"\nTotal Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
    
    # Show test details
    print(f"\nTest Results:")
    print("-" * 60)
    for result in results:
        status = "✓ PASS" if result['success'] else "✗ FAIL"
        print(f"{status} {result['name']:<35} {result['duration']:6.2f}s")
        if not result['success'] and result['error']:
            print(f"      Error: {result['error'][:100]}...")
    
    # Generate detailed report
    report_path = generate_report(results)
    print(f"\nDetailed report saved to: {report_path}")
    
    # Return exit code
    return 0 if failed_tests == 0 else 1

def generate_report(results):
    """Generate a detailed test report"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_path = f"sprint7_test_report_{timestamp}.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# FUA Sprint 7 Test Report\n\n")
        f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Summary
        total = len(results)
        passed = sum(1 for r in results if r['success'])
        failed = total - passed
        
        f.write("## Summary\n\n")
        f.write(f"- **Total Tests:** {total}\n")
        f.write(f"- **Passed:** {passed}\n")
        f.write(f"- **Failed:** {failed}\n")
        f.write(f"- **Success Rate:** {passed/total*100:.1f}%\n\n")
        
        # Test Results
        f.write("## Test Results\n\n")
        f.write("| Test | Status | Duration | Error |\n")
        f.write("|------|--------|----------|-------|\n")
        
        for result in results:
            status = "✓ PASS" if result['success'] else "✗ FAIL"
            error = result['error'][:100] + "..." if result['error'] else ""
            f.write(f"| {result['name']} | {status} | {result['duration']:.2f}s | {error} |\n")
        
        # Failed Tests Details
        if failed > 0:
            f.write("\n## Failed Test Details\n\n")
            for result in results:
                if not result['success']:
                    f.write(f"### {result['name']}\n\n")
                    f.write(f"**Duration:** {result['duration']:.2f}s\n\n")
                    f.write("**Error:**\n```\n")
                    f.write(result['error'])
                    f.write("\n```\n\n")
    
    return report_path

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)