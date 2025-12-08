#!/usr/bin/env python3
"""
Test JSON Logging
Tests JSON export structure and schema validation
"""

import sys
import logging
import json
import tempfile
from pathlib import Path

from src.metrics_collector import JSONExporter
import jsonschema

# Setup logging for tests
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def test_json_exporter_initialization():
    """Test JSONExporter initialization."""
    logger.info("Testing JSONExporter initialization...")

    # Test with custom temporary directory instead of hardcoded "results"
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.debug(f"Using temp directory: {temp_dir}")
        try:
            exporter = JSONExporter(temp_dir)
            assert exporter is not None, "JSONExporter should be initialized"
            assert exporter.output_dir == Path(temp_dir), (
                "Output dir should be set to temp directory"
            )
            assert exporter.output_dir.exists(), "Output directory should be created"
            logger.debug("JSONExporter initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize JSONExporter: {e}")
            raise

    logger.info("✓ JSONExporter initialization test passed")


def test_hierarchical_json_structure():
    """Test hierarchical JSON structure creation."""
    print("Testing hierarchical JSON structure...")

    exporter = JSONExporter()

    # Create test data
    test_simulations = [
        {
            "simulation_id": "test_sim_1",
            "controller": "CustomController",
            "location": "CA",
            "season": "summer",
            "week": 1,
            "battery_levels": [{"time": "2024-01-01 00:00:00", "level": 80.0}],
            "model_selections": {"YOLOv10_N": 10, "YOLOv10_M": 5},
            "model_energy_breakdown": {"YOLOv10_N": 0.001, "YOLOv10_M": 0.002},
            "total_tasks": 15,
            "completed_tasks": 14,
            "total_energy_wh": 0.003,
            "clean_energy_wh": 0.002,
            "clean_energy_percentage": 66.7,
            "success": True,
        },
        {
            "simulation_id": "test_sim_2",
            "controller": "NaiveWeakController",
            "location": "FL",
            "season": "winter",
            "week": 2,
            "battery_levels": [{"time": "2024-01-01 00:00:00", "level": 60.0}],
            "model_selections": {"YOLOv10_N": 20},
            "model_energy_breakdown": {"YOLOv10_N": 0.001},
            "total_tasks": 20,
            "completed_tasks": 18,
            "total_energy_wh": 0.001,
            "clean_energy_wh": 0.0005,
            "clean_energy_percentage": 50.0,
            "success": True,
        },
    ]

    test_aggregated = [
        {
            "controller": "CustomController",
            "location": "CA",
            "season": "summer",
            "total_simulations": 1,
            "successful_simulations": 1,
            "success_rate": 100.0,
            "avg_task_completion_rate": 93.3,
            "avg_clean_energy_percentage": 66.7,
            "total_energy_wh": 0.003,
            "clean_energy_wh": 0.002,
            "timestamp": "2024-01-01T00:00:00",
        }
    ]

    # Export to temporary file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=True
    ) as temp_file:
        temp_path = temp_file.name

        # Export data
        result_path = exporter.export_results(
            all_simulations=test_simulations,
            aggregated_data=test_aggregated,
            filename=Path(temp_path).name,
        )

        assert result_path != "", "Export should return valid path"
        assert Path(result_path).exists(), "Exported file should exist"

        # Load and validate structure
        with open(result_path, "r") as f:
            exported_data = json.load(f)

        # Validate top-level structure
        required_sections = [
            "metadata",
            "aggregated_metrics",
            "detailed_metrics",
            "time_series",
        ]
        for section in required_sections:
            assert section in exported_data, f"Missing required section: {section}"

        # Validate metadata
        metadata = exported_data["metadata"]
        assert "export_timestamp" in metadata, "Missing export_timestamp in metadata"
        assert "export_version" in metadata, "Missing export_version in metadata"
        assert "schema" in metadata, "Missing schema in metadata"

        # Validate aggregated metrics
        aggregated = exported_data["aggregated_metrics"]
        assert isinstance(aggregated, list), "Aggregated metrics should be a list"
        assert len(aggregated) == len(test_aggregated), (
            "Aggregated metrics length mismatch"
        )

        # Validate detailed metrics
        detailed = exported_data["detailed_metrics"]
        assert isinstance(detailed, list), "Detailed metrics should be a list"
        assert len(detailed) == len(test_simulations), (
            "Detailed metrics length mismatch"
        )

        # Check that time series data is extracted
        time_series = exported_data["time_series"]
        assert isinstance(time_series, dict), "Time series should be a dictionary"
        assert "battery_levels" in time_series, "Missing battery_levels in time_series"
        assert "model_selections" in time_series, (
            "Missing model_selections in time_series"
        )
        assert "energy_breakdown" in time_series, (
            "Missing energy_breakdown in time_series"
        )

        print("✓ Hierarchical JSON structure test passed")


def test_json_schema_validation():
    """Test JSON schema validation."""
    print("Testing JSON schema validation...")

    # Define expected schema
    expected_schema = {
        "type": "object",
        "required": [
            "metadata",
            "aggregated_metrics",
            "detailed_metrics",
            "time_series",
        ],
        "properties": {
            "metadata": {
                "type": "object",
                "required": ["export_timestamp", "export_version", "schema"],
                "properties": {
                    "export_timestamp": {"type": "string"},
                    "export_version": {"type": "string"},
                    "schema": {"type": "object"},
                },
            },
            "aggregated_metrics": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["controller", "location", "season", "success_rate"],
                },
            },
            "detailed_metrics": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["simulation_id", "controller", "location", "success"],
                },
            },
            "time_series": {
                "type": "object",
                "required": ["battery_levels", "model_selections", "energy_breakdown"],
                "properties": {
                    "battery_levels": {"type": "object"},
                    "model_selections": {"type": "object"},
                    "energy_breakdown": {"type": "object"},
                },
            },
        },
    }

    # Create test data and export
    exporter = JSONExporter()

    test_simulations = [
        {
            "simulation_id": "schema_test",
            "controller": "TestController",
            "location": "NY",
            "season": "spring",
            "week": 1,
            "battery_levels": [{"time": "2024-01-01 00:00:00", "level": 75.0}],
            "model_selections": {"YOLOv10_S": 8},
            "model_energy_breakdown": {"YOLOv10_S": 0.001},
            "total_tasks": 8,
            "completed_tasks": 8,
            "total_energy_wh": 0.001,
            "clean_energy_wh": 0.0008,
            "clean_energy_percentage": 80.0,
            "success": True,
        }
    ]

    test_aggregated = [
        {
            "controller": "TestController",
            "location": "NY",
            "season": "spring",
            "total_simulations": 1,
            "successful_simulations": 1,
            "success_rate": 100.0,
            "avg_task_completion_rate": 100.0,
            "avg_clean_energy_percentage": 80.0,
            "total_energy_wh": 0.001,
            "clean_energy_wh": 0.0008,
            "timestamp": "2024-01-01T00:00:00",
        }
    ]

    # Export and validate
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=True
    ) as temp_file:
        temp_path = temp_file.name

        result_path = exporter.export_results(
            all_simulations=test_simulations,
            aggregated_data=test_aggregated,
            filename=Path(temp_path).name,
        )

        # Load and validate against schema
        with open(result_path, "r") as f:
            exported_data = json.load(f)

        # Validate schema
        jsonschema.validate(exported_data, expected_schema)

        print("✓ JSON schema validation test passed")


def test_time_series_extraction():
    """Test time series data extraction."""
    print("Testing time series extraction...")

    exporter = JSONExporter()

    # Create test data with rich time series
    test_simulations = [
        {
            "simulation_id": "time_series_test",
            "controller": "CustomController",
            "location": "CA",
            "season": "summer",
            "week": 1,
            "battery_levels": [
                {"time": "2024-01-01 00:00:00", "level": 100.0},
                {"time": "2024-01-01 01:00:00", "level": 95.0},
                {"time": "2024-01-01 02:00:00", "level": 90.0},
            ],
            "model_selections": {"YOLOv10_N": 5, "YOLOv10_S": 3, "YOLOv10_M": 2},
            "model_energy_breakdown": {
                "YOLOv10_N": 0.0005,
                "YOLOv10_S": 0.0003,
                "YOLOv10_M": 0.0004,
            },
            "total_tasks": 10,
            "completed_tasks": 10,
            "total_energy_wh": 0.0012,
            "clean_energy_wh": 0.0010,
            "clean_energy_percentage": 83.3,
            "success": True,
        }
    ]

    test_aggregated = [
        {
            "controller": "CustomController",
            "location": "CA",
            "season": "summer",
            "total_simulations": 1,
            "successful_simulations": 1,
            "success_rate": 100.0,
            "avg_task_completion_rate": 100.0,
            "avg_clean_energy_percentage": 83.3,
            "total_energy_wh": 0.0012,
            "clean_energy_wh": 0.0010,
            "timestamp": "2024-01-01T00:00:00",
        }
    ]

    # Export and check time series
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=True
    ) as temp_file:
        temp_path = temp_file.name

        result_path = exporter.export_results(
            all_simulations=test_simulations,
            aggregated_data=test_aggregated,
            filename=Path(temp_path).name,
        )

        with open(result_path, "r") as f:
            exported_data = json.load(f)

        time_series = exported_data["time_series"]

        # Check battery levels
        battery_levels = time_series["battery_levels"]
        assert "time_series_test" in battery_levels, (
            "Missing simulation in battery levels"
        )

        sim_battery = battery_levels["time_series_test"]
        assert sim_battery["location"] == "CA", "Incorrect location in battery levels"
        assert sim_battery["season"] == "summer", "Incorrect season in battery levels"
        assert sim_battery["controller"] == "CustomController", (
            "Incorrect controller in battery levels"
        )
        assert len(sim_battery["levels"]) == 3, "Incorrect number of battery levels"

        # Check model selections
        model_selections = time_series["model_selections"]
        assert "time_series_test" in model_selections, (
            "Missing simulation in model selections"
        )

        sim_models = model_selections["time_series_test"]
        assert sim_models["selections"]["YOLOv10_N"] == 5, (
            "Incorrect model selection count"
        )
        assert sim_models["selections"]["YOLOv10_S"] == 3, (
            "Incorrect model selection count"
        )
        assert sim_models["selections"]["YOLOv10_M"] == 2, (
            "Incorrect model selection count"
        )

        # Check energy breakdown
        energy_breakdown = time_series["energy_breakdown"]
        assert "time_series_test" in energy_breakdown, (
            "Missing simulation in energy breakdown"
        )

        sim_energy = energy_breakdown["time_series_test"]
        assert sim_energy["breakdown"]["YOLOv10_N"] == 0.0005, (
            "Incorrect energy breakdown"
        )
        assert sim_energy["breakdown"]["YOLOv10_S"] == 0.0003, (
            "Incorrect energy breakdown"
        )
        assert sim_energy["breakdown"]["YOLOv10_M"] == 0.0004, (
            "Incorrect energy breakdown"
        )

        print("✓ Time series extraction test passed")


def test_empty_data_handling():
    """Test handling of empty data."""
    print("Testing empty data handling...")

    exporter = JSONExporter()

    # Test with empty data
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=True
    ) as temp_file:
        temp_path = temp_file.name

        result_path = exporter.export_results(
            all_simulations=[], aggregated_data=[], filename=Path(temp_path).name
        )

        # Should return empty string for no data
        assert result_path == "", "Export with no data should return empty path"

        print("✓ Empty data handling test passed")


def test_large_data_export():
    """Test export with large dataset."""
    print("Testing large data export...")

    exporter = JSONExporter()

    # Create larger test dataset
    large_simulations = []
    large_aggregated = []

    for i in range(100):  # 100 simulations
        sim = {
            "simulation_id": f"large_test_{i}",
            "controller": f"Controller_{i % 3}",
            "location": ["CA", "FL", "NY"][i % 3],
            "season": ["summer", "winter", "spring", "fall"][i % 4],
            "week": (i % 4) + 1,
            "battery_levels": [
                {"time": f"2024-01-01 {i:02d}:00:00", "level": 100.0 - i}
            ],
            "model_selections": {"YOLOv10_N": i + 1},
            "model_energy_breakdown": {"YOLOv10_N": 0.001 * (i + 1)},
            "total_tasks": i + 1,
            "completed_tasks": i + 1,
            "total_energy_wh": 0.001 * (i + 1),
            "clean_energy_wh": 0.0008 * (i + 1),
            "clean_energy_percentage": 80.0,
            "success": True,
        }
        large_simulations.append(sim)

    # Create aggregated data
    for controller in ["Controller_0", "Controller_1", "Controller_2"]:
        agg = {
            "controller": controller,
            "location": "CA",
            "season": "summer",
            "total_simulations": 34,
            "successful_simulations": 34,
            "success_rate": 100.0,
            "avg_task_completion_rate": 100.0,
            "avg_clean_energy_percentage": 80.0,
            "total_energy_wh": 0.001 * 34,
            "clean_energy_wh": 0.0008 * 34,
            "timestamp": "2024-01-01T00:00:00",
        }
        large_aggregated.append(agg)

    # Export large dataset
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=True
    ) as temp_file:
        temp_path = temp_file.name

        result_path = exporter.export_results(
            all_simulations=large_simulations,
            aggregated_data=large_aggregated,
            filename=Path(temp_path).name,
        )

        assert result_path != "", "Large dataset export should succeed"
        assert Path(result_path).exists(), "Exported file should exist"

        # Check file size (should be substantial)
        file_size = Path(result_path).stat().st_size
        assert file_size > 1000, (
            f"Exported file should be substantial: {file_size} bytes"
        )

        # Load and verify structure
        with open(result_path, "r") as f:
            exported_data = json.load(f)

        assert len(exported_data["detailed_metrics"]) == 100, (
            "Should have 100 detailed metrics"
        )
        assert len(exported_data["aggregated_metrics"]) == 3, (
            "Should have 3 aggregated metrics"
        )
        assert len(exported_data["time_series"]["battery_levels"]) == 100, (
            "Should have 100 battery time series"
        )

        print(f"✓ Large data export test passed ({file_size} bytes)")


def test_json_export_method():
    """Test individual JSON export method."""
    print("Testing individual JSON export method...")

    exporter = JSONExporter()

    # Test data
    test_data = {
        "test_key": "test_value",
        "nested_data": {"array": [1, 2, 3], "object": {"a": 1, "b": 2}},
        "number": 42,
        "boolean": True,
    }

    # Export using individual method
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=True
    ) as temp_file:
        temp_path = temp_file.name

        result_path = exporter.export_json(test_data, Path(temp_path).name)

        assert result_path != "", "Individual export should return valid path"
        assert Path(result_path).exists(), "Exported file should exist"

        # Load and verify
        with open(result_path, "r") as f:
            loaded_data = json.load(f)

        assert loaded_data == test_data, "Exported data should match original"

        print("✓ Individual JSON export method test passed")


def main():
    """Run all JSON logging tests."""
    logger.info("🧪 Running JSON Logging Tests")
    logger.info("=" * 50)

    try:
        test_json_exporter_initialization()
        test_hierarchical_json_structure()
        test_json_schema_validation()
        test_time_series_extraction()
        test_empty_data_handling()
        test_large_data_export()
        test_json_export_method()

        logger.info("\n✅ All JSON logging tests passed!")
        return 0

    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
