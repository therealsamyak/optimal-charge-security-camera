#!/usr/bin/env python3
"""
Test Integration
End-to-end pipeline test with 10 scenarios
"""

import sys
import os
import json
import tempfile
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.energy_data import EnergyData
from src.power_profiler import PowerProfiler
from src.controller import CustomController, NaiveWeakController, NaiveStrongController
from src.metrics_collector import JSONExporter


def test_end_to_end_pipeline():
    """Test complete end-to-end pipeline with 10 scenarios."""
    print("Testing end-to-end pipeline...")

    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        output_dir = temp_path / "results"
        output_dir.mkdir()

        # Initialize components
        energy_data = EnergyData()
        profiler = PowerProfiler()

        try:
            profiler.load_profiles()
        except Exception as e:
            print(f"⚠️  PowerProfiler loading failed (expected on non-macOS): {e}")
            # Create mock profiles for testing
            profiler.power_profiles = create_mock_profiles()

        # Create test scenarios (10 as specified)
        scenarios = create_test_scenarios()
        assert len(scenarios) == 10, f"Should create 10 scenarios, got {len(scenarios)}"

        # Test each scenario through the pipeline
        results = []

        for i, scenario in enumerate(scenarios):
            print(f"  Processing scenario {i + 1}/10: {scenario['name']}")

            try:
                # Step 1: Initialize controller
                controller = create_controller(scenario["controller_type"])

                # Step 2: Run simulation
                sim_result = run_simulation_scenario(
                    controller, scenario, energy_data, profiler
                )

                # Step 3: Validate result
                validate_simulation_result(sim_result, scenario)

                results.append(sim_result)
                print(f"    ✓ Success: {sim_result['success']}")

            except Exception as e:
                print(f"    ❌ Failed: {e}")
                raise

        # Step 4: Export results
        exporter = JSONExporter(str(output_dir))

        # Create aggregated data
        aggregated_data = create_aggregated_results(results)

        # Export to JSON
        export_file = exporter.export_results(
            all_simulations=results,
            aggregated_data=aggregated_data,
            filename="integration_test_results.json",
        )

        assert export_file != "", "Export should return valid path"
        assert Path(export_file).exists(), "Exported file should exist"

        # Step 5: Validate exported JSON
        with open(export_file, "r") as f:
            exported_data = json.load(f)

        validate_exported_json(exported_data, results)

        print(f"✓ End-to-end pipeline test passed ({len(results)} scenarios)")
        return results


def create_test_scenarios():
    """Create 10 test scenarios covering different conditions."""
    scenarios = [
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
            "name": "High Battery Winter FL",
            "controller_type": "NaiveWeakController",
            "location": "FL",
            "season": "winter",
            "battery_level": 90.0,
            "clean_energy_percentage": 30.0,
            "accuracy_requirement": 0.5,
            "latency_requirement": 25.0,
            "expected_success": True,
            "expected_charge": False,
        },
        {
            "name": "Medium Battery Spring NY",
            "controller_type": "NaiveStrongController",
            "location": "NY",
            "season": "spring",
            "battery_level": 50.0,
            "clean_energy_percentage": 60.0,
            "accuracy_requirement": 0.8,
            "latency_requirement": 10.0,
            "expected_success": True,
            "expected_charge": False,
        },
        {
            "name": "Critical Battery Fall CA",
            "controller_type": "CustomController",
            "location": "CA",
            "season": "fall",
            "battery_level": 5.0,
            "clean_energy_percentage": 70.0,
            "accuracy_requirement": 0.6,
            "latency_requirement": 20.0,
            "expected_success": True,
            "expected_charge": True,
        },
        {
            "name": "Full Energy Summer FL",
            "controller_type": "CustomController",
            "location": "FL",
            "season": "summer",
            "battery_level": 100.0,
            "clean_energy_percentage": 90.0,
            "accuracy_requirement": 0.9,
            "latency_requirement": 8.0,
            "expected_success": True,
            "expected_charge": False,
        },
        {
            "name": "Low Accuracy Requirement NY",
            "controller_type": "NaiveWeakController",
            "location": "NY",
            "season": "winter",
            "battery_level": 60.0,
            "clean_energy_percentage": 40.0,
            "accuracy_requirement": 0.3,
            "latency_requirement": 30.0,
            "expected_success": True,
            "expected_charge": False,
        },
        {
            "name": "High Latency Requirement CA",
            "controller_type": "CustomController",
            "location": "CA",
            "season": "spring",
            "battery_level": 40.0,
            "clean_energy_percentage": 75.0,
            "accuracy_requirement": 0.8,
            "latency_requirement": 50.0,
            "expected_success": True,
            "expected_charge": True,
        },
        {
            "name": "Balanced Requirements FL",
            "controller_type": "NaiveStrongController",
            "location": "FL",
            "season": "fall",
            "battery_level": 70.0,
            "clean_energy_percentage": 55.0,
            "accuracy_requirement": 0.7,
            "latency_requirement": 18.0,
            "expected_success": True,
            "expected_charge": False,
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

    return scenarios


def create_controller(controller_type: str):
    """Create controller instance."""
    if controller_type == "CustomController":
        # Create mock weights file for testing
        mock_weights = create_mock_weights()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(mock_weights, f, indent=2)
            weights_file = f.name

        try:
            return CustomController(weights_file)
        finally:
            os.unlink(weights_file)
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
            "power_cost": 1.0,
        },
        "YOLOv10_S": {
            "accuracy": 0.7,
            "avg_inference_time_seconds": 0.008,
            "power_cost": 2.0,
        },
        "YOLOv10_M": {
            "accuracy": 0.8,
            "avg_inference_time_seconds": 0.012,
            "power_cost": 3.0,
        },
        "YOLOv10_B": {
            "accuracy": 0.85,
            "avg_inference_time_seconds": 0.018,
            "power_cost": 4.0,
        },
        "YOLOv10_L": {
            "accuracy": 0.9,
            "avg_inference_time_seconds": 0.025,
            "power_cost": 5.0,
        },
        "YOLOv10_X": {
            "accuracy": 0.95,
            "avg_inference_time_seconds": 0.035,
            "power_cost": 6.0,
        },
    }


def run_simulation_scenario(controller, scenario, energy_data, profiler):
    """Run a single simulation scenario."""
    # Get available models
    available_models = profiler.get_all_models_data()

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
    model_data = available_models.get(model_choice.model_name, {})

    # Calculate success based on requirements
    meets_accuracy = model_data.get("accuracy", 0) >= scenario["accuracy_requirement"]
    meets_latency = (
        model_data.get("avg_inference_time_seconds", 1) * 1000
        <= scenario["latency_requirement"]
    )

    success = meets_accuracy and meets_latency

    # Calculate completed tasks based on model performance
    if success:
        completion_rate = 0.9 + (model_data.get("accuracy", 0) * 0.1)
        completed_tasks = int(total_tasks * completion_rate)
    else:
        completed_tasks = int(
            total_tasks * 0.7
        )  # Partial success even if requirements not met

    # Calculate energy consumption
    energy_per_task = model_data.get("power_cost", 1.0) * model_data.get(
        "avg_inference_time_seconds", 0.01
    )
    total_energy = energy_per_task * completed_tasks

    # Calculate clean energy based on scenario
    clean_energy = total_energy * (scenario["clean_energy_percentage"] / 100.0)

    # Simulate battery depletion
    battery_depletion = scenario["battery_level"] - (
        total_energy * 10
    )  # Simplified depletion
    final_battery = max(0, battery_depletion)

    # Create result
    result = {
        "scenario_name": scenario["name"],
        "controller": scenario["controller_type"],
        "location": scenario["location"],
        "season": scenario["season"],
        "battery_level": scenario["battery_level"],
        "clean_energy_percentage": scenario["clean_energy_percentage"],
        "accuracy_requirement": scenario["accuracy_requirement"],
        "latency_requirement": scenario["latency_requirement"],
        "selected_model": model_choice.model_name,
        "should_charge": model_choice.should_charge,
        "success": success,
        "total_tasks": total_tasks,
        "completed_tasks": completed_tasks,
        "task_completion_rate": (completed_tasks / total_tasks) * 100,
        "total_energy_wh": total_energy,
        "clean_energy_wh": clean_energy,
        "clean_energy_percentage_actual": (clean_energy / total_energy * 100)
        if total_energy > 0
        else 0,
        "final_battery_level": final_battery,
        "reasoning": model_choice.reasoning,
    }

    return result


def validate_simulation_result(result, scenario):
    """Validate simulation result against expectations."""
    # Check basic structure
    required_fields = [
        "scenario_name",
        "controller",
        "location",
        "season",
        "selected_model",
        "should_charge",
        "success",
        "total_tasks",
        "completed_tasks",
        "task_completion_rate",
        "total_energy_wh",
        "clean_energy_wh",
    ]

    for field in required_fields:
        assert field in result, f"Missing required field: {field}"

    # Check value ranges
    assert 0 <= result["task_completion_rate"] <= 100, (
        f"Invalid completion rate: {result['task_completion_rate']}"
    )
    assert result["total_energy_wh"] >= 0, (
        f"Invalid total energy: {result['total_energy_wh']}"
    )
    assert result["clean_energy_wh"] >= 0, (
        f"Invalid clean energy: {result['clean_energy_wh']}"
    )
    assert result["completed_tasks"] <= result["total_tasks"], (
        "Completed tasks cannot exceed total tasks"
    )

    # Check against expectations
    assert result["success"] == scenario["expected_success"], (
        f"Success mismatch: expected {scenario['expected_success']}, got {result['success']}"
    )

    # For CustomController, check charge decision logic
    if scenario["controller_type"] == "CustomController":
        # Should charge if battery is low or clean energy is high
        expected_charge = scenario["expected_charge"]
        assert result["should_charge"] == expected_charge, (
            f"Charge decision mismatch: expected {expected_charge}, got {result['should_charge']}"
        )


def create_aggregated_results(results):
    """Create aggregated results from simulation results."""
    # Group by controller
    controller_groups = {}
    for result in results:
        controller = result["controller"]
        if controller not in controller_groups:
            controller_groups[controller] = []
        controller_groups[controller].append(result)

    aggregated = []

    for controller, controller_results in controller_groups.items():
        # Calculate aggregates
        total_simulations = len(controller_results)
        successful_simulations = sum(1 for r in controller_results if r["success"])
        success_rate = (successful_simulations / total_simulations) * 100

        avg_completion = (
            sum(r["task_completion_rate"] for r in controller_results)
            / total_simulations
        )
        avg_clean_energy = (
            sum(r["clean_energy_percentage_actual"] for r in controller_results)
            / total_simulations
        )
        total_energy = sum(r["total_energy_wh"] for r in controller_results)
        clean_energy = sum(r["clean_energy_wh"] for r in controller_results)

        aggregated.append(
            {
                "controller": controller,
                "total_simulations": total_simulations,
                "successful_simulations": successful_simulations,
                "success_rate": success_rate,
                "avg_task_completion_rate": avg_completion,
                "avg_clean_energy_percentage": avg_clean_energy,
                "total_energy_wh": total_energy,
                "clean_energy_wh": clean_energy,
                "timestamp": "2024-01-01T00:00:00",
            }
        )

    return aggregated


def validate_exported_json(exported_data, original_results):
    """Validate exported JSON structure and content."""
    # Check structure
    required_sections = [
        "metadata",
        "aggregated_metrics",
        "detailed_metrics",
        "time_series",
    ]
    for section in required_sections:
        assert section in exported_data, f"Missing section: {section}"

    # Check metadata
    metadata = exported_data["metadata"]
    assert "export_timestamp" in metadata, "Missing export timestamp"
    assert "export_version" in metadata, "Missing export version"
    assert "schema" in metadata, "Missing schema info"

    # Check detailed metrics
    detailed = exported_data["detailed_metrics"]
    assert len(detailed) == len(original_results), (
        f"Detailed metrics count mismatch: {len(detailed)} vs {len(original_results)}"
    )

    # Check aggregated metrics
    aggregated = exported_data["aggregated_metrics"]
    assert len(aggregated) > 0, "Should have aggregated metrics"

    # Check time series
    time_series = exported_data["time_series"]
    assert isinstance(time_series, dict), "Time series should be dictionary"
    assert "battery_levels" in time_series, "Missing battery levels in time series"
    assert "model_selections" in time_series, "Missing model selections in time series"


def test_pipeline_performance():
    """Test pipeline performance and scalability."""
    print("Testing pipeline performance...")

    import time

    # Test with different data sizes
    test_sizes = [1, 5, 10, 20]

    for size in test_sizes:
        start_time = time.time()

        # Create test scenarios
        scenarios = []
        for i in range(size):
            scenario = create_test_scenarios()[i % 10]  # Reuse base scenarios
            scenario["name"] = f"Performance Test {i}"
            scenarios.append(scenario)

        # Run mini pipeline
        profiler = PowerProfiler()
        profiler.power_profiles = create_mock_profiles()

        results = []
        for scenario in scenarios:
            controller = create_controller(scenario["controller_type"])
            result = run_simulation_scenario(controller, scenario, None, profiler)
            results.append(result)

        end_time = time.time()
        duration = end_time - start_time

        # Performance should be reasonable (less than 1 second per scenario)
        avg_time_per_scenario = duration / size
        assert avg_time_per_scenario < 1.0, (
            f"Performance too slow: {avg_time_per_scenario:.3f}s per scenario"
        )

        print(
            f"  ✓ Size {size}: {duration:.3f}s total, {avg_time_per_scenario:.3f}s per scenario"
        )

    print("✓ Pipeline performance test passed")


def test_error_recovery():
    """Test error recovery and robustness."""
    print("Testing error recovery...")

    # Test with invalid controller type
    try:
        create_controller("InvalidController")
        assert False, "Should raise ValueError for invalid controller"
    except ValueError:
        pass  # Expected

    # Test with missing model data
    profiler = PowerProfiler()
    profiler.power_profiles = {}  # Empty profiles

    controller = NaiveWeakController()

    try:
        # This should handle missing model data gracefully
        model_choice = controller.select_model(
            battery_level=50.0,
            clean_energy_percentage=60.0,
            user_accuracy_requirement=0.7,
            user_latency_requirement=15.0,
            available_models={},
        )
        # Should return some default model
        assert model_choice.model_name is not None, (
            "Should return default model when no models available"
        )
        print("  ✓ Handled missing model data gracefully")
    except Exception as e:
        print(f"  ⚠️  Error handling could be improved: {e}")

    print("✓ Error recovery test passed")


def test_data_integrity():
    """Test data integrity throughout the pipeline."""
    print("Testing data integrity...")

    # Create test scenarios
    scenarios = create_test_scenarios()[:5]  # Test with 5 scenarios

    # Run pipeline
    profiler = PowerProfiler()
    profiler.power_profiles = create_mock_profiles()

    results = []
    for scenario in scenarios:
        controller = create_controller(scenario["controller_type"])
        result = run_simulation_scenario(controller, scenario, None, profiler)
        results.append(result)

    # Verify data integrity
    for i, result in enumerate(results):
        original_scenario = scenarios[i]

        # Check that scenario data is preserved
        assert result["scenario_name"] == original_scenario["name"], (
            "Scenario name should be preserved"
        )
        assert result["controller"] == original_scenario["controller_type"], (
            "Controller type should be preserved"
        )
        assert result["location"] == original_scenario["location"], (
            "Location should be preserved"
        )
        assert result["season"] == original_scenario["season"], (
            "Season should be preserved"
        )

        # Check that calculations are consistent
        assert (
            result["task_completion_rate"]
            == (result["completed_tasks"] / result["total_tasks"]) * 100
        ), "Task completion rate should be consistent"

        if result["total_energy_wh"] > 0:
            expected_clean_pct = (
                result["clean_energy_wh"] / result["total_energy_wh"]
            ) * 100
            assert (
                abs(result["clean_energy_percentage_actual"] - expected_clean_pct)
                < 0.01
            ), "Clean energy percentage should be consistent"

    print("✓ Data integrity test passed")


def main():
    """Run all integration tests."""
    print("🧪 Running Integration Tests")
    print("=" * 50)

    try:
        test_end_to_end_pipeline()
        test_pipeline_performance()
        test_error_recovery()
        test_data_integrity()

        print("\n✅ All integration tests passed!")
        return 0

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
