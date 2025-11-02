"""
Test runner script for OCS Camera.

This script runs all tests and provides a summary.
"""

import sys
import os
import subprocess
import time

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def run_test_file(test_file):
    """Run a single test file and return results."""
    print(f"\n{'=' * 60}")
    print(f"Running {test_file}")
    print("=" * 60)

    start_time = time.time()

    try:
        # Run the test file
        result = subprocess.run(
            [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(__file__),
        )

        duration = time.time() - start_time

        print(f"Duration: {duration:.2f}s")
        print(f"Exit code: {result.returncode}")

        if result.stdout:
            print("STDOUT:")
            print(result.stdout)

        if result.stderr:
            print("STDERR:")
            print(result.stderr)

        return result.returncode == 0, duration

    except Exception as e:
        print(f"Error running {test_file}: {e}")
        return False, 0


def main():
    """Run all tests and provide summary."""
    print("OCS Camera Test Suite")
    print("=" * 60)

    # List of test files to run
    test_files = [
        "test_mock_sensors.py",
        "test_config.py",
        "test_controller.py",
        "test_utils.py",
        "test_integration.py",
    ]

    results = []
    total_duration = 0

    for test_file in test_files:
        test_path = os.path.join(os.path.dirname(__file__), test_file)
        if os.path.exists(test_path):
            success, duration = run_test_file(test_file)
            results.append((test_file, success))
            total_duration += duration
        else:
            print(f"\nTest file not found: {test_file}")
            results.append((test_file, False))

    # Summary
    print(f"\n{'=' * 60}")
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    print(f"Total tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Total duration: {total_duration:.2f}s")

    print("\nDetailed Results:")
    for test_file, success in results:
        status = "PASS" if success else "FAIL"
        print(f"  {test_file}: {status}")

    # Return appropriate exit code
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
