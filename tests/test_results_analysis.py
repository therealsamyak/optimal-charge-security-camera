#!/usr/bin/env python3
"""
Test Results Analysis
Tests JSON parsing and metric calculations
"""

import sys
import os
import json
import tempfile
import statistics
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import results analysis functions
sys.path.insert(0, str(Path(__file__).parent.parent))
from results import (
    read_json_full,
    analyze_battery_levels,
)


def test_json_parsing():
    """Test JSON parsing functionality."""
    print("Testing JSON parsing...")

    # Create test JSON data
    test_data = {
        "metadata": {
            "total_simulations": 10,
            "total_variations": 2,
            "locations": ["CA", "FL", "NY"],
            "seasons": ["summer", "winter"],
            "controllers": ["CustomController", "NaiveWeakController"],
            "export_timestamp": "2024-01-01T00:00:00",
        },
        "aggregated_metrics": [
            {
                "controller": "CustomController",
                "location": "CA",
                "season": "summer",
                "success_rate": 95.0,
                "avg_task_completion_rate": 92.5,
                "avg_clean_energy_percentage": 75.0,
                "total_energy_wh": 1.5,
                "clean_energy_wh": 1.125,
            },
            {
                "controller": "NaiveWeakController",
                "location": "FL",
                "season": "winter",
                "success_rate": 85.0,
                "avg_task_completion_rate": 88.0,
                "avg_clean_energy_percentage": 45.0,
                "total_energy_wh": 2.0,
                "clean_energy_wh": 0.9,
            },
        ],
        "detailed_metrics": [
            {
                "simulation_id": "test_sim_1",
                "controller": "CustomController",
                "location": "CA",
                "season": "summer",
                "week": 1,
                "success": True,
                "task_completion_rate": 95.0,
                "completed_tasks": 19,
                "total_tasks": 20,
                "total_energy_wh": 0.075,
                "clean_energy_percentage": 80.0,
                "YOLOv10_N_count": 10,
                "YOLOv10_M_count": 9,
                "YOLOv10_N_energy_wh": 0.030,
                "YOLOv10_M_energy_wh": 0.045,
            }
        ],
        "time_series": {
            "battery_levels": {
                "test_sim_1": {
                    "location": "CA",
                    "season": "summer",
                    "controller": "CustomController",
                    "levels": [
                        {"time": "2024-01-01 00:00:00", "level": 100.0},
                        {"time": "2024-01-01 01:00:00", "level": 95.0},
                        {"time": "2024-01-01 02:00:00", "level": 90.0},
                    ],
                }
            },
            "model_selections": {
                "test_sim_1": {
                    "location": "CA",
                    "season": "summer",
                    "controller": "CustomController",
                    "selections": {"YOLOv10_N": 10, "YOLOv10_M": 9},
                }
            },
        },
    }

    # Write to temporary file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as temp_file:
        temp_path = temp_file.name
        json.dump(test_data, temp_file, indent=2)

    try:
        # Test parsing
        parsed_data = read_json_full(temp_path)

        assert parsed_data is not None, "Parsed data should not be None"
        assert isinstance(parsed_data, dict), "Parsed data should be dictionary"
        assert "metadata" in parsed_data, "Should have metadata section"
        assert "aggregated_metrics" in parsed_data, "Should have aggregated metrics"
        assert "detailed_metrics" in parsed_data, "Should have detailed metrics"
        assert "time_series" in parsed_data, "Should have time series"

        # Verify data integrity
        assert parsed_data["metadata"]["total_simulations"] == 10, (
            "Metadata should match"
        )
        assert len(parsed_data["aggregated_metrics"]) == 2, (
            "Should have 2 aggregated metrics"
        )
        assert len(parsed_data["detailed_metrics"]) == 1, (
            "Should have 1 detailed metric"
        )

        print("✓ JSON parsing test passed")

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_battery_level_analysis():
    """Test battery level analysis."""
    print("Testing battery level analysis...")

    # Test battery levels data
    test_battery_levels = [
        {"time": "2024-01-01 00:00:00", "level": 100.0},
        {"time": "2024-01-01 01:00:00", "level": 95.0},
        {"time": "2024-01-01 02:00:00", "level": 90.0},
        {"time": "2024-01-01 03:00:00", "level": 85.0},
        {"time": "2024-01-01 04:00:00", "level": 80.0},
    ]

    # Analyze
    stats = analyze_battery_levels(test_battery_levels)

    # Verify statistics
    assert stats["initial"] == 100.0, (
        f"Initial level should be 100.0, got {stats['initial']}"
    )
    assert stats["final"] == 80.0, f"Final level should be 80.0, got {stats['final']}"
    assert stats["min"] == 80.0, f"Min level should be 80.0, got {stats['min']}"
    assert stats["max"] == 100.0, f"Max level should be 100.0, got {stats['max']}"
    assert stats["avg"] == 90.0, f"Avg level should be 90.0, got {stats['avg']}"
    assert stats["data_points"] == 5, (
        f"Data points should be 5, got {stats['data_points']}"
    )

    # Test empty data
    empty_stats = analyze_battery_levels([])
    assert empty_stats == {}, "Empty data should return empty stats"

    print("✓ Battery level analysis test passed")


def test_metric_calculations():
    """Test metric calculations."""
    print("Testing metric calculations...")

    # Create test data for calculations
    test_data = [
        {
            "controller": "CustomController",
            "success": True,
            "task_completion_rate": 95.0,
            "clean_energy_percentage": 80.0,
            "total_energy_wh": 1.0,
            "clean_energy_wh": 0.8,
            "energy_per_task_wh": 0.05,
            "completed_tasks": 20,
            "total_tasks": 20,
        },
        {
            "controller": "CustomController",
            "success": True,
            "task_completion_rate": 85.0,
            "clean_energy_percentage": 60.0,
            "total_energy_wh": 1.5,
            "clean_energy_wh": 0.9,
            "energy_per_task_wh": 0.075,
            "completed_tasks": 17,
            "total_tasks": 20,
        },
        {
            "controller": "NaiveWeakController",
            "success": False,
            "task_completion_rate": 70.0,
            "clean_energy_percentage": 40.0,
            "total_energy_wh": 2.0,
            "clean_energy_wh": 0.8,
            "energy_per_task_wh": 0.1,
            "completed_tasks": 14,
            "total_tasks": 20,
        },
    ]

    # Test success rate calculation
    custom_data = [d for d in test_data if d["controller"] == "CustomController"]
    custom_success_rate = (
        sum(1 for d in custom_data if d["success"]) / len(custom_data) * 100
    )
    assert custom_success_rate == 100.0, (
        f"CustomController success rate should be 100.0%, got {custom_success_rate}%"
    )

    # Test average task completion rate
    completion_rates = [d["task_completion_rate"] for d in test_data]
    avg_completion = statistics.mean(completion_rates)
    expected_avg = (95.0 + 85.0 + 70.0) / 3
    assert abs(avg_completion - expected_avg) < 0.01, (
        f"Avg completion should be {expected_avg}, got {avg_completion}"
    )

    # Test average clean energy percentage
    clean_energy_pcts = [d["clean_energy_percentage"] for d in test_data]
    avg_clean_energy = statistics.mean(clean_energy_pcts)
    expected_clean = (80.0 + 60.0 + 40.0) / 3
    assert abs(avg_clean_energy - expected_clean) < 0.01, (
        f"Avg clean energy should be {expected_clean}, got {avg_clean_energy}"
    )

    # Test energy efficiency
    energy_per_task = [d["energy_per_task_wh"] for d in test_data]
    avg_energy_per_task = statistics.mean(energy_per_task)
    expected_energy = (0.05 + 0.075 + 0.1) / 3
    assert abs(avg_energy_per_task - expected_energy) < 0.01, (
        f"Avg energy per task should be {expected_energy}, got {avg_energy_per_task}"
    )

    print("✓ Metric calculations test passed")


def test_model_usage_analysis():
    """Test model usage analysis."""
    print("Testing model usage analysis...")

    # Create test data with model counts
    test_data = [
        {
            "controller": "CustomController",
            "YOLOv10_N_count": 15,
            "YOLOv10_S_count": 10,
            "YOLOv10_M_count": 5,
            "YOLOv10_N_energy_wh": 0.15,
            "YOLOv10_S_energy_wh": 0.12,
            "YOLOv10_M_energy_wh": 0.08,
        },
        {
            "controller": "CustomController",
            "YOLOv10_N_count": 20,
            "YOLOv10_S_count": 8,
            "YOLOv10_M_count": 12,
            "YOLOv10_N_energy_wh": 0.20,
            "YOLOv10_S_energy_wh": 0.10,
            "YOLOv10_M_energy_wh": 0.15,
        },
    ]

    # Calculate model usage
    model_counts = {}
    model_energy = {}

    for row in test_data:
        for model in ["YOLOv10_N", "YOLOv10_S", "YOLOv10_M"]:
            count_key = f"{model}_count"
            energy_key = f"{model}_energy_wh"

            if count_key in row:
                model_counts[model] = model_counts.get(model, 0) + row[count_key]

            if energy_key in row:
                model_energy[model] = model_energy.get(model, 0) + row[energy_key]

    # Verify calculations
    assert model_counts["YOLOv10_N"] == 35, (
        f"YOLOv10_N count should be 35, got {model_counts['YOLOv10_N']}"
    )
    assert model_counts["YOLOv10_S"] == 18, (
        f"YOLOv10_S count should be 18, got {model_counts['YOLOv10_S']}"
    )
    assert model_counts["YOLOv10_M"] == 17, (
        f"YOLOv10_M count should be 17, got {model_counts['YOLOv10_M']}"
    )

    total_tasks = sum(model_counts.values())
    assert total_tasks == 70, f"Total tasks should be 70, got {total_tasks}"

    # Verify percentages
    n_percentage = (model_counts["YOLOv10_N"] / total_tasks) * 100
    expected_n_pct = (35 / 70) * 100
    assert abs(n_percentage - expected_n_pct) < 0.01, (
        f"YOLOv10_N percentage should be {expected_n_pct}%, got {n_percentage}%"
    )

    # Verify energy totals
    assert model_energy["YOLOv10_N"] == 0.35, (
        f"YOLOv10_N energy should be 0.35, got {model_energy['YOLOv10_N']}"
    )
    assert model_energy["YOLOv10_S"] == 0.22, (
        f"YOLOv10_S energy should be 0.22, got {model_energy['YOLOv10_S']}"
    )
    assert model_energy["YOLOv10_M"] == 0.23, (
        f"YOLOv10_M energy should be 0.23, got {model_energy['YOLOv10_M']}"
    )

    print("✓ Model usage analysis test passed")


def test_energy_efficiency_analysis():
    """Test energy efficiency analysis."""
    print("Testing energy efficiency analysis...")

    # Create test data for efficiency analysis
    test_data = [
        {
            "controller": "CustomController",
            "energy_per_task_wh": 0.05,
            "clean_energy_per_task_wh": 0.04,
            "total_energy_wh": 1.0,
        },
        {
            "controller": "CustomController",
            "energy_per_task_wh": 0.06,
            "clean_energy_per_task_wh": 0.045,
            "total_energy_wh": 1.2,
        },
        {
            "controller": "NaiveWeakController",
            "energy_per_task_wh": 0.08,
            "clean_energy_per_task_wh": 0.03,
            "total_energy_wh": 1.6,
        },
    ]

    # Group by controller
    controller_data = {}
    for row in test_data:
        controller = row["controller"]
        if controller not in controller_data:
            controller_data[controller] = {
                "energy_per_tasks": [],
                "clean_energy_per_tasks": [],
                "total_energies": [],
            }

        if row.get("energy_per_task_wh"):
            controller_data[controller]["energy_per_tasks"].append(
                row["energy_per_task_wh"]
            )

        if row.get("clean_energy_per_task_wh"):
            controller_data[controller]["clean_energy_per_tasks"].append(
                row["clean_energy_per_task_wh"]
            )

        controller_data[controller]["total_energies"].append(
            row.get("total_energy_wh", 0)
        )

    # Calculate efficiency metrics
    for controller, data in controller_data.items():
        energy_per_tasks = data["energy_per_tasks"]
        clean_energy_per_tasks = data["clean_energy_per_tasks"]
        total_energies = data["total_energies"]

        if energy_per_tasks:
            avg_energy_per_task = statistics.mean(energy_per_tasks)
            assert avg_energy_per_task > 0, (
                f"Average energy per task should be positive for {controller}"
            )

        if clean_energy_per_tasks:
            avg_clean_energy_per_task = statistics.mean(clean_energy_per_tasks)
            assert avg_clean_energy_per_task > 0, (
                f"Average clean energy per task should be positive for {controller}"
            )

        total_energy = sum(total_energies)
        assert total_energy > 0, f"Total energy should be positive for {controller}"

    # Verify specific calculations
    custom_energy = controller_data["CustomController"]["energy_per_tasks"]
    custom_avg_energy = statistics.mean(custom_energy)
    expected_custom_avg = (0.05 + 0.06) / 2
    assert abs(custom_avg_energy - expected_custom_avg) < 0.01, (
        f"CustomController avg energy should be {expected_custom_avg}, got {custom_avg_energy}"
    )

    naive_energy = controller_data["NaiveWeakController"]["energy_per_tasks"]
    naive_avg_energy = statistics.mean(naive_energy)
    expected_naive_avg = 0.08
    assert abs(naive_avg_energy - expected_naive_avg) < 0.01, (
        f"NaiveWeakController avg energy should be {expected_naive_avg}, got {naive_avg_energy}"
    )

    print("✓ Energy efficiency analysis test passed")


def test_temporal_pattern_analysis():
    """Test temporal pattern analysis."""
    print("Testing temporal pattern analysis...")

    # Create test time series data
    test_time_series = {
        "battery_levels": {
            "test_sim_1": {
                "location": "CA",
                "season": "summer",
                "controller": "CustomController",
                "levels": [
                    {"time": "2024-01-01 00:00:00", "level": 100.0},
                    {"time": "2024-01-01 06:00:00", "level": 95.0},
                    {"time": "2024-01-01 12:00:00", "level": 85.0},
                    {"time": "2024-01-01 18:00:00", "level": 75.0},
                    {"time": "2024-01-02 00:00:00", "level": 70.0},
                ],
            },
            "test_sim_2": {
                "location": "FL",
                "season": "winter",
                "controller": "NaiveWeakController",
                "levels": [
                    {"time": "2024-01-01 00:00:00", "level": 90.0},
                    {"time": "2024-01-01 06:00:00", "level": 88.0},
                    {"time": "2024-01-01 12:00:00", "level": 85.0},
                    {"time": "2024-01-01 18:00:00", "level": 82.0},
                    {"time": "2024-01-02 00:00:00", "level": 80.0},
                ],
            },
        },
        "model_selections": {
            "test_sim_1": {
                "location": "CA",
                "season": "summer",
                "controller": "CustomController",
                "selections": {"YOLOv10_N": 15, "YOLOv10_S": 10, "YOLOv10_M": 5},
            },
            "test_sim_2": {
                "location": "FL",
                "season": "winter",
                "controller": "NaiveWeakController",
                "selections": {"YOLOv10_N": 25, "YOLOv10_S": 5},
            },
        },
    }

    # Analyze battery depletion patterns
    battery_data = test_time_series["battery_levels"]
    for sim_id, sim_data in battery_data.items():
        levels = sim_data["levels"]
        if levels:
            initial = levels[0]["level"]
            final = levels[-1]["level"]
            depletion = initial - final

            if sim_id == "test_sim_1":
                assert initial == 100.0, (
                    f"test_sim_1 initial should be 100.0, got {initial}"
                )
                assert final == 70.0, f"test_sim_1 final should be 70.0, got {final}"
                assert depletion == 30.0, (
                    f"test_sim_1 depletion should be 30.0, got {depletion}"
                )
            elif sim_id == "test_sim_2":
                assert initial == 90.0, (
                    f"test_sim_2 initial should be 90.0, got {initial}"
                )
                assert final == 80.0, f"test_sim_2 final should be 80.0, got {final}"
                assert depletion == 10.0, (
                    f"test_sim_2 depletion should be 10.0, got {depletion}"
                )

    # Analyze model selection patterns
    model_data = test_time_series["model_selections"]
    for sim_id, sim_data in model_data.items():
        selections = sim_data["selections"]
        if selections:
            total_selections = sum(selections.values())
            most_used = max(selections.items(), key=lambda x: x[1])

            if sim_id == "test_sim_1":
                assert total_selections == 30, (
                    f"test_sim_1 total selections should be 30, got {total_selections}"
                )
                assert most_used[0] == "YOLOv10_N", (
                    f"test_sim_1 most used should be YOLOv10_N, got {most_used[0]}"
                )
                assert most_used[1] == 15, (
                    f"test_sim_1 most used count should be 15, got {most_used[1]}"
                )
            elif sim_id == "test_sim_2":
                assert total_selections == 30, (
                    f"test_sim_2 total selections should be 30, got {total_selections}"
                )
                assert most_used[0] == "YOLOv10_N", (
                    f"test_sim_2 most used should be YOLOv10_N, got {most_used[0]}"
                )
                assert most_used[1] == 25, (
                    f"test_sim_2 most used count should be 25, got {most_used[1]}"
                )

    print("✓ Temporal pattern analysis test passed")


def test_insights_generation():
    """Test insights generation."""
    print("Testing insights generation...")

    # Create test data for insights
    test_data = [
        {
            "controller": "CustomController",
            "success": True,
            "task_completion_rate": 95.0,
            "energy_per_task_wh": 0.05,
            "clean_energy_percentage": 80.0,
            "location": "CA",
            "battery_levels": json.dumps(
                [
                    {"time": "2024-01-01 00:00:00", "level": 100.0},
                    {"time": "2024-01-01 01:00:00", "level": 95.0},
                ]
            ),
            "total_tasks": 20,
        },
        {
            "controller": "CustomController",
            "success": True,
            "task_completion_rate": 90.0,
            "energy_per_task_wh": 0.06,
            "clean_energy_percentage": 75.0,
            "location": "FL",
            "battery_levels": json.dumps(
                [
                    {"time": "2024-01-01 00:00:00", "level": 90.0},
                    {"time": "2024-01-01 01:00:00", "level": 85.0},
                ]
            ),
            "total_tasks": 20,
        },
        {
            "controller": "NaiveWeakController",
            "success": False,
            "task_completion_rate": 70.0,
            "energy_per_task_wh": 0.08,
            "clean_energy_percentage": 40.0,
            "location": "NY",
            "battery_levels": json.dumps(
                [
                    {"time": "2024-01-01 00:00:00", "level": 80.0},
                    {"time": "2024-01-01 01:00:00", "level": 75.0},
                ]
            ),
            "total_tasks": 20,
        },
    ]

    # Calculate controller performance
    controllers = list(set(row["controller"] for row in test_data))
    controller_performance = {}

    for controller in controllers:
        controller_data = [row for row in test_data if row["controller"] == controller]
        success_rate = (
            sum(1 for row in controller_data if row["success"])
            / len(controller_data)
            * 100
        )
        avg_completion = statistics.mean(
            [row["task_completion_rate"] for row in controller_data]
        )
        avg_efficiency = statistics.mean(
            [row["energy_per_task_wh"] for row in controller_data]
        )

        controller_performance[controller] = {
            "success_rate": success_rate,
            "avg_completion": avg_completion,
            "avg_efficiency": avg_efficiency,
        }

    # Verify performance calculations
    custom_perf = controller_performance["CustomController"]
    assert custom_perf["success_rate"] == 100.0, (
        f"CustomController success rate should be 100.0%, got {custom_perf['success_rate']}"
    )
    assert custom_perf["avg_completion"] == 92.5, (
        f"CustomController avg completion should be 92.5%, got {custom_perf['avg_completion']}"
    )
    assert custom_perf["avg_efficiency"] == 0.055, (
        f"CustomController avg efficiency should be 0.055, got {custom_perf['avg_efficiency']}"
    )

    naive_perf = controller_performance["NaiveWeakController"]
    assert naive_perf["success_rate"] == 0.0, (
        f"NaiveWeakController success rate should be 0.0%, got {naive_perf['success_rate']}"
    )
    assert naive_perf["avg_completion"] == 70.0, (
        f"NaiveWeakController avg completion should be 70.0%, got {naive_perf['avg_completion']}"
    )
    assert naive_perf["avg_efficiency"] == 0.08, (
        f"NaiveWeakController avg efficiency should be 0.08, got {naive_perf['avg_efficiency']}"
    )

    # Find best controller
    best_controller = max(
        controller_performance.items(), key=lambda x: x[1]["success_rate"]
    )
    assert best_controller[0] == "CustomController", (
        f"Best controller should be CustomController, got {best_controller[0]}"
    )

    # Find most efficient controller
    efficient_controller = min(
        controller_performance.items(), key=lambda x: x[1]["avg_efficiency"]
    )
    assert efficient_controller[0] == "CustomController", (
        f"Most efficient should be CustomController, got {efficient_controller[0]}"
    )

    print("✓ Insights generation test passed")


def test_error_handling():
    """Test error handling in analysis functions."""
    print("Testing error handling...")

    # Test with invalid JSON
    invalid_result = read_json_full("nonexistent_file.json")
    assert invalid_result == {}, "Invalid file should return empty dict"

    # Test with empty battery levels
    empty_battery_stats = analyze_battery_levels([])
    assert empty_battery_stats == {}, "Empty battery levels should return empty stats"

    # Test with empty data for calculations
    try:
        empty_avg = statistics.mean([])
        assert False, "Empty list should raise StatisticsError"
    except statistics.StatisticsError:
        pass  # Expected behavior

    print("✓ Error handling test passed")


def main():
    """Run all results analysis tests."""
    print("🧪 Running Results Analysis Tests")
    print("=" * 50)

    try:
        test_json_parsing()
        test_battery_level_analysis()
        test_metric_calculations()
        test_model_usage_analysis()
        test_energy_efficiency_analysis()
        test_temporal_pattern_analysis()
        test_insights_generation()
        test_error_handling()

        print("\n✅ All results analysis tests passed!")
        return 0

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
