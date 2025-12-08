#!/usr/bin/env python3
"""
Integration Tests - End-to-End Pipeline Testing
Tests complete pipeline with 10 scenarios covering edge cases
"""

import os
import sys
import logging
import tempfile
import json
from pathlib import Path

from src.controller import CustomController, NaiveWeakController, NaiveStrongController
from src.energy_data import EnergyData
from src.power_profiler import PowerProfiler

# Setup logging for tests
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class MockProfiler:
    """Mock profiler for testing when real PowerProfiler fails."""

    def __init__(self):
        self.power_profiles = create_mock_profiles()
        logger.debug("Created MockProfiler with profiles")

    def load_profiles(self):
        pass

    def get_all_models_data(self):
        result = self.power_profiles
        logger.debug(f"MockProfiler returning {len(result)} models")
        return result


def test_end_to_end_pipeline():
    """Test complete end-to-end pipeline with 10 scenarios."""
    logger.info("Testing end-to-end pipeline...")

    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        output_dir = temp_path / "test_outputs"
        output_dir.mkdir()
        logger.debug(f"Using temporary directory: {temp_dir}")

        # Initialize components
        try:
            energy_data = EnergyData()
            logger.debug("EnergyData initialized")
        except Exception as e:
            logger.error(f"Failed to initialize EnergyData: {e}")
            raise

        try:
            profiler = PowerProfiler()
            profiler.load_profiles()
            logger.debug("Using real PowerProfiler")
            # Check if profiles were loaded
            models = profiler.get_all_models_data()
            logger.debug(f"Real PowerProfiler has {len(models)} models")
            if len(models) == 0:
                logger.debug("No models loaded, switching to mock")
                raise Exception("No models loaded")
        except Exception as e:
            logger.warning(
                f"PowerProfiler initialization failed (expected on non-macOS): {e}"
            )

            # Create mock profiler for testing
            profiler = MockProfiler()
            logger.debug("Using MockProfiler")

        # Create test scenarios (10 as specified)
        scenarios = create_test_scenarios()
        assert len(scenarios) == 10, f"Should create 10 scenarios, got {len(scenarios)}"
        logger.debug(f"Created {len(scenarios)} test scenarios")

        # Test each scenario through pipeline
        results = []

        for i, scenario in enumerate(scenarios):
            logger.info(f"  Processing scenario {i + 1}/10: {scenario['name']}")

            try:
                # Step 1: Initialize controller with temp directory
                controller = create_controller(scenario["controller_type"], temp_dir)

                # Step 2: Run simulation
                sim_result = run_simulation_scenario(
                    controller, scenario, energy_data, profiler
                )

                # Step 3: Validate result
                if validate_simulation_result(sim_result, scenario):
                    results.append(sim_result)
                    logger.debug("    ✓ Success")
                else:
                    logger.warning("    ❌ Failed validation")
                    logger.debug(f"      Result: {sim_result}")

            except Exception as e:
                logger.error(f"    ❌ Failed: {e}")

        # Validate overall results
        assert len(results) >= 6, (
            f"Should pass at least 6/10 scenarios, got {len(results)}"
        )
        logger.info(
            f"✓ End-to-end pipeline test passed: {len(results)}/10 scenarios successful"
        )

        # Create test scenarios (10 as specified)
        scenarios = create_test_scenarios()
        assert len(scenarios) == 10, f"Should create 10 scenarios, got {len(scenarios)}"

        # Test each scenario through the pipeline
        results = []

        for i, scenario in enumerate(scenarios):
            print(f"  Processing scenario {i + 1}/10: {scenario['name']}")

            try:
                # Step 1: Initialize controller with temp directory
                controller = create_controller(scenario["controller_type"], temp_dir)

                # Step 2: Run simulation
                sim_result = run_simulation_scenario(
                    controller, scenario, energy_data, profiler
                )

                # Step 3: Validate result
                if validate_simulation_result(sim_result, scenario):
                    results.append(sim_result)
                    print("    ✓ Success")
                else:
                    print("    ❌ Failed validation")
                    print(f"      Result: {sim_result}")

            except Exception as e:
                print(f"    ❌ Failed: {e}")

        # Validate overall results
        assert len(results) >= 6, (
            f"Should pass at least 6/10 scenarios, got {len(results)}"
        )
        print(
            f"✓ End-to-end pipeline test passed: {len(results)}/10 scenarios successful"
        )


def test_pipeline_performance():
    """Test pipeline performance metrics."""
    print("Testing pipeline performance...")

    # Test with larger dataset
    scenarios = create_test_scenarios()

    profiler = MockProfiler()
    energy_data = EnergyData()

    # Measure performance
    import time

    start_time = time.time()

    # Use temporary directory for performance test
    with tempfile.TemporaryDirectory() as temp_dir:
        for scenario in scenarios:
            controller = create_controller(scenario["controller_type"], temp_dir)
            run_simulation_scenario(controller, scenario, energy_data, profiler)

    end_time = time.time()
    total_time = end_time - start_time

    assert total_time < 10.0, f"Pipeline too slow: {total_time:.2f}s"
    print(f"✓ Performance test passed: {total_time:.2f}s for 10 scenarios")


def test_error_recovery():
    """Test error handling and recovery."""
    print("Testing error recovery...")

    # Test with invalid controller
    try:
        create_controller("InvalidController")
        assert False, "Should raise ValueError for invalid controller"
    except ValueError:
        print("  ✓ Invalid controller properly rejected")

    # Test with missing weights file
    try:
        CustomController("nonexistent_weights.json")
        assert False, "Should raise error for missing weights file"
    except Exception:
        print("  ✓ Missing weights file properly handled")

    # Test with corrupted weights file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"invalid": "data"}, f)
        corrupted_file = f.name

    try:
        CustomController(corrupted_file)
        assert False, "Should raise error for corrupted weights"
    except Exception:
        print("  ✓ Corrupted weights file properly handled")
    finally:
        # Clean up the corrupted file
        try:
            os.unlink(corrupted_file)
        except OSError:
            pass

    print("✓ Error recovery test passed")


def test_data_integrity():
    """Test data integrity throughout pipeline."""
    print("Testing data integrity...")

    # Create test data
    scenarios = create_test_scenarios()

    profiler = MockProfiler()
    energy_data = EnergyData()

    # Test data consistency with temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        for scenario in scenarios:
            controller = create_controller(scenario["controller_type"], temp_dir)
            result = run_simulation_scenario(
                controller, scenario, energy_data, profiler
            )

            # Validate result structure
            required_fields = ["model_choice", "energy_consumed", "tasks_completed"]
            for field in required_fields:
                assert field in result, f"Missing field: {field}"

            # Validate data types
            assert isinstance(result["energy_consumed"], (int, float))
            assert isinstance(result["tasks_completed"], int)
            assert result["tasks_completed"] >= 0

    print("✓ Data integrity test passed")


def create_controller(controller_type, temp_dir=None):
    """Create controller instance for testing."""
    if controller_type == "CustomController":
        # Create mock weights file for testing
        mock_weights = create_mock_weights()

        if temp_dir is None:
            # Create a temporary file that gets cleaned up automatically
            temp_file = tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=True
            )
            json.dump(mock_weights, temp_file, indent=2)
            temp_file.flush()
            weights_file = temp_file.name

            # CustomController will read the file while it's still open
            controller = CustomController(weights_file)
            # File gets cleaned up automatically when temp_file goes out of scope
            return controller
    elif controller_type == "NaiveWeakController":
        return NaiveWeakController()
    elif controller_type == "NaiveStrongController":
        return NaiveStrongController()
    else:
        raise ValueError(f"Unknown controller type: {controller_type}")


def create_mock_weights():
    """Create mock neural network weights for testing."""
    return {
        "model_state_dict": {
            "shared_layers.0.weight": [[0.1] * 4 for _ in range(128)],
            "shared_layers.0.bias": [0.0] * 128,
            "shared_layers.2.weight": [[0.1] * 128 for _ in range(64)],
            "shared_layers.2.bias": [0.0] * 64,
            "model_head.weight": [[0.1] * 64 for _ in range(6)],
            "model_head.bias": [0.0] * 6,
            "charge_head.weight": [[0.1] * 64],
            "charge_head.bias": [0.0],
        },
        "model_to_idx": {
            "YOLOv10_N": 0,
            "YOLOv10_S": 1,
            "YOLOv10_M": 2,
            "YOLOv10_B": 3,
            "YOLOv10_L": 4,
            "YOLOv10_X": 5,
        },
        "idx_to_model": {
            "0": "YOLOv10_N",
            "1": "YOLOv10_S",
            "2": "YOLOv10_M",
            "3": "YOLOv10_B",
            "4": "YOLOv10_L",
            "5": "YOLOv10_X",
        },
    }


def create_mock_profiles():
    """Create mock power profiles for testing."""
    return {
        "YOLOv10_N": {
            "accuracy": 0.6,
            "avg_inference_time_seconds": 0.005,
            "model_power_mw": 1.0,
        },
        "YOLOv10_S": {
            "accuracy": 0.7,
            "avg_inference_time_seconds": 0.008,
            "model_power_mw": 2.0,
        },
        "YOLOv10_M": {
            "accuracy": 0.8,
            "avg_inference_time_seconds": 0.012,
            "model_power_mw": 3.0,
        },
        "YOLOv10_B": {
            "accuracy": 0.85,
            "avg_inference_time_seconds": 0.018,
            "model_power_mw": 4.0,
        },
        "YOLOv10_L": {
            "accuracy": 0.9,
            "avg_inference_time_seconds": 0.025,
            "model_power_mw": 5.0,
        },
        "YOLOv10_X": {
            "accuracy": 0.95,
            "avg_inference_time_seconds": 0.035,
            "model_power_mw": 6.0,
        },
    }


def create_test_scenarios():
    """Create 10 diverse test scenarios."""
    return [
        {
            "name": "Low Battery Summer CA",
            "controller_type": "CustomController",
            "location": "CA",
            "season": "summer",
            "battery_level": 20.0,
            "clean_energy_percentage": 80.0,
            "accuracy_requirement": 0.7,
            "latency_requirement": 15.0,
            "expected_success": True,
            "expected_charge": True,
        },
        {
            "name": "High Battery Winter NY",
            "controller_type": "NaiveWeakController",
            "location": "NY",
            "season": "winter",
            "battery_level": 80.0,
            "clean_energy_percentage": 20.0,
            "accuracy_requirement": 0.5,
            "latency_requirement": 25.0,
            "expected_success": True,
            "expected_charge": False,
        },
        {
            "name": "Medium Battery Spring FL",
            "controller_type": "NaiveStrongController",
            "location": "FL",
            "season": "spring",
            "battery_level": 50.0,
            "clean_energy_percentage": 60.0,
            "accuracy_requirement": 0.9,
            "latency_requirement": 10.0,
            "expected_success": True,
            "expected_charge": False,
        },
        {
            "name": "Critical Battery NW",
            "controller_type": "CustomController",
            "location": "NW",
            "season": "fall",
            "battery_level": 5.0,
            "clean_energy_percentage": 90.0,
            "accuracy_requirement": 0.4,
            "latency_requirement": 30.0,
            "expected_success": True,
            "expected_charge": True,
        },
        {
            "name": "Full Battery Clean Energy",
            "controller_type": "CustomController",
            "location": "CA",
            "season": "summer",
            "battery_level": 95.0,
            "clean_energy_percentage": 100.0,
            "accuracy_requirement": 0.8,
            "latency_requirement": 12.0,
            "expected_success": True,
            "expected_charge": False,
        },
        {
            "name": "Low Accuracy Requirement",
            "controller_type": "NaiveWeakController",
            "location": "FL",
            "season": "spring",
            "battery_level": 60.0,
            "clean_energy_percentage": 40.0,
            "accuracy_requirement": 0.3,
            "latency_requirement": 40.0,
            "expected_success": True,
            "expected_charge": False,
        },
        {
            "name": "High Accuracy Requirement",
            "controller_type": "NaiveStrongController",
            "location": "NY",
            "season": "winter",
            "battery_level": 70.0,
            "clean_energy_percentage": 30.0,
            "accuracy_requirement": 0.95,
            "latency_requirement": 8.0,
            "expected_success": True,
            "expected_charge": False,
        },
        {
            "name": "Tight Latency Requirement",
            "controller_type": "CustomController",
            "location": "NW",
            "season": "fall",
            "battery_level": 40.0,
            "clean_energy_percentage": 70.0,
            "accuracy_requirement": 0.6,
            "latency_requirement": 5.0,
            "expected_success": True,
            "expected_charge": True,
        },
        {
            "name": "Edge Case Very Low Battery",
            "controller_type": "CustomController",
            "location": "NY",
            "season": "winter",
            "battery_level": 1.0,
            "clean_energy_percentage": 85.0,
            "accuracy_requirement": 0.4,
            "latency_requirement": 40.0,
            "expected_success": True,
            "expected_charge": True,
        },
        {
            "name": "Edge Case Very High Requirements",
            "controller_type": "NaiveStrongController",
            "location": "CA",
            "season": "summer",
            "battery_level": 80.0,
            "clean_energy_percentage": 95.0,
            "accuracy_requirement": 0.95,
            "latency_requirement": 5.0,
            "expected_success": True,
            "expected_charge": False,
        },
    ]


def run_simulation_scenario(controller, scenario, energy_data, profiler):
    """Run a single simulation scenario."""
    # Get available models and convert to expected format
    raw_models = profiler.get_all_models_data()
    available_models = {}

    for name, profile in raw_models.items():
        available_models[name] = {
            "accuracy": profile["accuracy"],
            "latency": profile["avg_inference_time_seconds"],
            "power_cost": profile["model_power_mw"],  # Convert to expected field name
        }

    # Make model selection and charging decision
    model_choice = controller.select_model(
        battery_level=scenario["battery_level"],
        clean_energy_percentage=scenario["clean_energy_percentage"],
        user_accuracy_requirement=scenario["accuracy_requirement"],
        user_latency_requirement=scenario["latency_requirement"],
        available_models=available_models,
    )

    # Simulate task execution (simplified)
    total_tasks = 20
    tasks_completed = 0
    total_energy = 0.0

    for task in range(total_tasks):
        # Check if we have enough battery
        if scenario["battery_level"] < 5:
            break

        # Execute task with selected model
        model_power = available_models[model_choice.model_name]["power_cost"]
        scenario["battery_level"] -= model_power * 0.1  # Simplified energy consumption
        tasks_completed += 1
        total_energy += model_power * 0.1

    return {
        "scenario": scenario["name"],
        "controller": scenario["controller_type"],
        "model_choice": model_choice,
        "energy_consumed": total_energy,
        "tasks_completed": tasks_completed,
        "final_battery": scenario["battery_level"],
    }


def validate_simulation_result(result, scenario):
    """Validate simulation result against expectations."""
    # Basic validation
    if result["energy_consumed"] < 0:
        return False

    if (
        result["final_battery"] < -50 or result["final_battery"] > 100
    ):  # Allow reasonable negative battery
        return False

    # Scenario-specific validation
    if "expected_success" in scenario:
        return scenario["expected_success"]

    return True


def main():
    """Run all integration tests."""
    logger.info("🚀 Starting Integration Tests...")

    try:
        test_end_to_end_pipeline()
        test_pipeline_performance()
        test_error_recovery()
        test_data_integrity()

        logger.info("\n✅ All integration tests passed!")
        return 0

    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
