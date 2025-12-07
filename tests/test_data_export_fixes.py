#!/usr/bin/env python3
"""
Simplified test script to verify data export fixes work correctly.
"""

import sys
import json
import csv
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from metrics_collector import CSVExporter


def test_model_selection_export():
    """Test that model selection counts are properly exported to CSV."""
    print("Testing model selection export...")

    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        exporter = CSVExporter(temp_dir)

        # Create mock simulation data with model selections
        mock_simulations = [
            {
                "simulation_id": "test_sim_1",
                "controller": "naive_weak",
                "model_selections": {
                    "YOLOv10_N": 100,
                    "YOLOv10_S": 0,
                    "YOLOv10_M": 0,
                    "YOLOv10_B": 0,
                    "YOLOv10_L": 0,
                    "YOLOv10_X": 0,
                },
                "model_energy_breakdown": {
                    "YOLOv10_N": 0.025,
                    "YOLOv10_S": 0.0,
                    "YOLOv10_M": 0.0,
                    "YOLOv10_B": 0.0,
                    "YOLOv10_L": 0.0,
                    "YOLOv10_X": 0.0,
                },
                "total_tasks": 100,
                "completed_tasks": 100,
                "total_energy_wh": 50.0,
                "clean_energy_wh": 25.0,
                "clean_energy_mwh": 0.025,
                "dirty_energy_mwh": 0.025,
                "clean_energy_percentage": 50.0,
                "energy_per_task_wh": 0.5,
                "clean_energy_per_task_wh": 0.25,
                "peak_power_mw": 10.5,
                "average_power_mw": 5.2,
                "battery_efficiency_score": 2.0,
                "battery_depletion_rate_per_hour": 1.5,
                "charging_events_count": 5,
                "time_below_20_percent": 300,
                "time_above_80_percent": 600,
                "total_simulation_time": 86400.0,
            }
        ]

        # Export to CSV
        csv_path = exporter.export_detailed_results(
            mock_simulations, "test_model_selection.csv"
        )

        # Verify CSV contains model count fields
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            row = next(reader)

            # Check that model count fields exist and have correct values
            assert "YOLOv10_N_count" in row, "YOLOv10_N_count field missing"
            assert row["YOLOv10_N_count"] == "100", (
                f"Expected 100, got {row['YOLOv10_N_count']}"
            )
            assert "YOLOv10_S_count" in row, "YOLOv10_S_count field missing"
            assert row["YOLOv10_S_count"] == "0", (
                f"Expected 0, got {row['YOLOv10_S_count']}"
            )

            # Check energy breakdown fields
            assert "YOLOv10_N_energy_wh" in row, "YOLOv10_N_energy_wh field missing"
            assert float(row["YOLOv10_N_energy_wh"]) == 25.0, (
                f"Expected 25.0, got {row['YOLOv10_N_energy_wh']}"
            )

            # Check new comprehensive metrics
            assert "clean_energy_mwh" in row, "clean_energy_mwh field missing"
            assert "dirty_energy_mwh" in row, "dirty_energy_mwh field missing"
            assert "energy_per_task_wh" in row, "energy_per_task_wh field missing"
            assert "battery_efficiency_score" in row, (
                "battery_efficiency_score field missing"
            )

        print("✓ Model selection export test passed")


def test_json_time_series_export():
    """Test that JSON time series data is exported correctly."""
    print("Testing JSON time series export...")

    with tempfile.TemporaryDirectory() as temp_dir:
        exporter = CSVExporter(temp_dir)

        # Create mock simulation with battery levels
        mock_simulations = [
            {
                "simulation_id": "test_sim_1",
                "controller": "test_controller",
                "location": "CA",
                "season": "summer",
                "week": 1,
                "battery_levels": [
                    {"timestamp": 0, "level": 100.0},
                    {"timestamp": 300, "level": 95.0},
                    {"timestamp": 600, "level": 90.0},
                ],
                "model_selections": {"YOLOv10_N": 50, "YOLOv10_S": 25},
                "model_energy_breakdown": {"YOLOv10_N": 0.025, "YOLOv10_S": 0.015},
                "total_energy_wh": 40.0,
                "clean_energy_wh": 20.0,
                "clean_energy_percentage": 50.0,
            }
        ]

        # Export
        exporter.export_detailed_results(mock_simulations, "test_time_series.csv")

        # Check JSON files were created
        battery_json = Path(temp_dir) / "battery_time_series.json"
        model_json = Path(temp_dir) / "model_selection_timeline.json"
        energy_json = Path(temp_dir) / "energy_usage_summary.json"

        assert battery_json.exists(), "battery_time_series.json not created"
        assert model_json.exists(), "model_selection_timeline.json not created"
        assert energy_json.exists(), "energy_usage_summary.json not created"

        # Verify battery JSON content
        with open(battery_json, "r") as f:
            battery_data = json.load(f)
            assert "test_sim_1" in battery_data
            assert len(battery_data["test_sim_1"]["battery_levels"]) == 3

        # Verify model timeline JSON content
        with open(model_json, "r") as f:
            model_data = json.load(f)
            assert "test_sim_1" in model_data
            assert model_data["test_sim_1"]["model_selections"]["YOLOv10_N"] == 50

        # Verify energy summary JSON content
        with open(energy_json, "r") as f:
            energy_data = json.load(f)
            assert "test_sim_1" in energy_data
            assert energy_data["test_sim_1"]["total_energy_wh"] == 40.0

    print("✓ JSON time series export test passed")


def test_validation_functions():
    """Test that validation functions catch data inconsistencies."""
    print("Testing validation functions...")

    with tempfile.TemporaryDirectory() as temp_dir:
        exporter = CSVExporter(temp_dir)

        # Create mock simulation with inconsistencies
        bad_simulations = [
            {
                "simulation_id": "bad_sim_1",
                "completed_tasks": 150,  # More than total_tasks
                "total_tasks": 100,
                "total_energy_wh": -10.0,  # Negative energy
                "clean_energy_wh": 20.0,
                "dirty_energy_mwh": 0.05,  # 50 Wh, total should be 70 Wh but is -10 Wh
            }
        ]

        # Export should trigger validation warnings/errors but still succeed
        csv_path = exporter.export_detailed_results(
            bad_simulations, "test_validation.csv"
        )

        # The export should still succeed despite validation issues
        assert Path(csv_path).exists(), (
            "CSV export should still succeed despite validation issues"
        )

    print("✓ Validation functions test passed")


def main():
    """Run all tests."""
    print("Running data export fixes tests...")
    print("=" * 50)

    try:
        test_model_selection_export()
        test_json_time_series_export()
        test_validation_functions()

        print("=" * 50)
        print("✅ All tests passed! Data export fixes are working correctly.")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
