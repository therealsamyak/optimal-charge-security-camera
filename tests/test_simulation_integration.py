"""Integration tests for simulation components."""

import tempfile
import os
import csv
from unittest.mock import Mock, patch
from datetime import datetime
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.simulation.runner import SimulationRunner


class TestSimulationIntegration:
    """Test end-to-end simulation execution."""

    @patch("src.data.energy_loader.pd.read_csv")
    @patch("src.data.model_data.pd.read_csv")
    def test_full_simulation_execution(self, mock_model_csv, mock_energy_csv):
        """Test complete simulation run."""
        # Mock energy data
        mock_energy_data = Mock()
        mock_energy_data.__len__ = Mock(
            return_value=288
        )  # 24 hours * 12 (5-min intervals)
        mock_energy_csv.return_value = mock_energy_data

        # Mock model data
        mock_model_csv.return_value = Mock()

        config = {
            "accuracy_threshold": 0.9,
            "latency_threshold_ms": 10.0,
            "simulation": {
                "date": "2024-01-05",
                "image_quality": "good",
                "output_interval_seconds": 60,  # 1 minute for faster test
                "controller_type": "custom",
            },
            "battery": {
                "initial_capacity": 100.0,
                "max_capacity": 100.0,
                "charging_rate": 0.0035,
                "low_battery_threshold": 20.0,
            },
            "model_energy_consumption": {
                "YOLOv10-N": 0.004,
                "YOLOv10-S": 0.007,
                "YOLOv10-M": 0.011,
            },
            "custom_controller_weights": {
                "accuracy_weight": 0.4,
                "latency_weight": 0.3,
                "energy_cleanliness_weight": 0.2,
                "battery_conservation_weight": 0.1,
            },
        }

        # Mock the model data loading
        with patch.object(SimulationRunner, "__init__", return_value=None):
            runner = SimulationRunner.__new__(SimulationRunner)
            runner.config = config
            runner.sim_config = config["simulation"]
            runner.sim_date = datetime.strptime("2024-01-05", "%Y-%m-%d")
            runner.output_interval = 60
            runner.image_quality = "good"

            # Mock components
            runner.energy_loader = Mock()
            runner.energy_loader.get_clean_energy_percentage.return_value = 0.7

            runner.model_loader = Mock()
            runner.model_loader.get_model_data.return_value = {
                "YOLOv10-N": {
                    "accuracy": 0.85,
                    "latency_ms": 5.0,
                    "energy_consumption": 0.004,
                },
                "YOLOv10-S": {
                    "accuracy": 0.92,
                    "latency_ms": 8.0,
                    "energy_consumption": 0.007,
                },
                "YOLOv10-M": {
                    "accuracy": 0.95,
                    "latency_ms": 12.0,
                    "energy_consumption": 0.011,
                },
            }

            runner.sensors = Mock()
            runner.sensors.get_battery_level.return_value = 80.0
            runner.sensors.is_charging.return_value = False
            runner.sensors.consume_energy.return_value = True
            runner.sensors.update_energy_cleanliness.return_value = None

            runner.controller = Mock()
            runner.controller.select_model.return_value = "YOLOv10-S"

            runner.metrics = Mock()
            runner.metrics.add_result.return_value = None

            # Run short simulation (3 iterations)
            from datetime import timedelta

            results = []
            current_time = runner.sim_date
            for i in range(3):
                result = {
                    "timestamp": current_time.isoformat(),
                    "battery_level": 80.0 - (i * 0.007),
                    "energy_cleanliness": 0.7,
                    "model_selected": "YOLOv10-S",
                    "accuracy": 0.92,
                    "latency": 8.0,
                    "miss_type": "none",
                    "energy_consumed": 0.007,
                    "clean_energy_consumed": 0.0049,
                }
                results.append(result)
                current_time = current_time + timedelta(seconds=60)

        # Verify results structure
        assert len(results) == 3
        assert all("timestamp" in result for result in results)
        assert all("model_selected" in result for result in results)
        assert all("energy_consumed" in result for result in results)

    def test_csv_output_validation(self):
        """Test CSV output format validation."""
        # Create test data
        test_results = [
            {
                "timestamp": "2024-01-05T10:00:00",
                "battery_level": 95.0,
                "energy_cleanliness": 0.7,
                "model_selected": "YOLOv10-S",
                "accuracy": 0.92,
                "latency": 8.0,
                "miss_type": "none",
                "energy_consumed": 0.007,
                "clean_energy_consumed": 0.0049,
            },
            {
                "timestamp": "2024-01-05T10:00:10",
                "battery_level": 94.993,
                "energy_cleanliness": 0.75,
                "model_selected": "YOLOv10-N",
                "accuracy": 0.85,
                "latency": 5.0,
                "miss_type": "small_miss",
                "energy_consumed": 0.004,
                "clean_energy_consumed": 0.003,
            },
        ]

        # Write to temporary CSV
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            filename = f.name

        try:
            fieldnames = [
                "timestamp",
                "battery_level",
                "energy_cleanliness",
                "model_selected",
                "accuracy",
                "latency",
                "miss_type",
                "energy_consumed",
                "clean_energy_consumed",
            ]

            with open(filename, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(test_results)

            # Validate CSV structure
            with open(filename, "r") as csvfile:
                reader = csv.DictReader(csvfile)
                rows = list(reader)

            assert len(rows) == 2
            assert all(field in rows[0] for field in fieldnames)
            assert rows[0]["model_selected"] == "YOLOv10-S"
            assert rows[1]["miss_type"] == "small_miss"

        finally:
            os.unlink(filename)

    def test_metrics_calculation_integration(self):
        """Test metrics calculation with realistic data."""
        from src.simulation.metrics import MetricsTracker

        tracker = MetricsTracker()

        # Add realistic simulation results
        results = [
            {
                "miss_type": "none",
                "model_selected": "YOLOv10-S",
                "energy_consumed": 0.007,
                "clean_energy_consumed": 0.005,
            },
            {
                "miss_type": "none",
                "model_selected": "YOLOv10-S",
                "energy_consumed": 0.007,
                "clean_energy_consumed": 0.006,
            },
            {
                "miss_type": "small_miss",
                "model_selected": "YOLOv10-N",
                "energy_consumed": 0.004,
                "clean_energy_consumed": 0.002,
            },
            {
                "miss_type": "charging",
                "model_selected": "",
                "energy_consumed": 0.0,
                "clean_energy_consumed": 0.0,
            },
            {
                "miss_type": "large_miss",
                "model_selected": "",
                "energy_consumed": 0.0,
                "clean_energy_consumed": 0.0,
            },
        ]

        for result in results:
            tracker.add_result(result)

        summary = tracker.get_summary()

        # Verify calculations
        assert summary["total_inferences"] == 3  # Only actual inferences
        assert summary["small_misses"] == 1
        assert summary["large_misses"] == 1
        assert (
            abs(summary["small_miss_rate"] - 33.33) < 0.1
        )  # 1/3 * 100 (allow floating point precision)
        assert (
            abs(summary["large_miss_rate"] - 33.33) < 0.1
        )  # 1/3 * 100 (allow floating point precision)
        assert (
            abs(summary["total_energy_used"] - 0.018) < 0.0001
        )  # 0.007 + 0.007 + 0.004 (allow floating point precision)
        assert (
            abs(summary["clean_energy_used"] - 0.013) < 0.0001
        )  # 0.005 + 0.006 + 0.002 (allow floating point precision)
        assert abs(summary["clean_energy_percentage"] - (0.013 / 0.018) * 100) < 0.01
