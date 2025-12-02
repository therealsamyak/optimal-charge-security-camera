#!/usr/bin/env python3
"""
Unit tests for MetricsCollector and CSVExporter classes.
"""

import unittest
import tempfile
import json
import csv
from pathlib import Path

from src.metrics_collector import MetricsCollector, CSVExporter


class TestMetricsCollector(unittest.TestCase):
    """Unit tests for MetricsCollector class."""

    def setUp(self):
        """Set up test fixtures."""
        self.collector = MetricsCollector()

    def test_metrics_initialization(self):
        """Test that metrics are properly initialized."""
        metrics = self.collector.get_metrics()

        # Check initial values
        self.assertEqual(metrics["total_tasks"], 0)
        self.assertEqual(metrics["completed_tasks"], 0)
        self.assertEqual(metrics["missed_deadlines"], 0)
        self.assertEqual(metrics["total_energy_wh"], 0.0)
        self.assertEqual(metrics["clean_energy_wh"], 0.0)
        self.assertEqual(metrics["task_completion_rate"], 0.0)
        self.assertEqual(metrics["clean_energy_percentage"], 0.0)

    def test_task_metrics_update(self):
        """Test task metrics update functionality."""
        # Simulate completed task
        task_data = {
            "completed": True,
            "energy_used_wh": 0.001,
            "clean_energy_used_wh": 0.0008,
            "model_used": "YOLOv10_N",
        }

        self.collector.update_task_metrics(task_data)
        metrics = self.collector.get_metrics()

        self.assertEqual(metrics["total_tasks"], 1)
        self.assertEqual(metrics["completed_tasks"], 1)
        self.assertEqual(metrics["total_energy_wh"], 0.001)
        self.assertEqual(metrics["clean_energy_wh"], 0.0008)
        self.assertEqual(metrics["model_selections"]["YOLOv10_N"], 1)
        self.assertEqual(metrics["small_model_tasks"], 1)

    def test_missed_task_metrics(self):
        """Test metrics for missed tasks."""
        # Simulate missed task
        task_data = {"completed": False, "model_used": "YOLOv10_X"}

        self.collector.update_task_metrics(task_data)
        metrics = self.collector.get_metrics()

        self.assertEqual(metrics["total_tasks"], 1)
        self.assertEqual(metrics["completed_tasks"], 0)
        self.assertEqual(metrics["missed_deadlines"], 1)
        self.assertEqual(metrics["large_model_misses"], 1)

    def test_battery_level_tracking(self):
        """Test battery level tracking."""
        # Add some battery level readings
        self.collector.update_battery_level(0.0, 100.0)
        self.collector.update_battery_level(5.0, 95.0)
        self.collector.update_battery_level(10.0, 90.0)

        metrics = self.collector.get_metrics()

        self.assertEqual(len(metrics["battery_levels"]), 3)
        self.assertEqual(metrics["battery_levels"][0]["timestamp"], 0.0)
        self.assertEqual(metrics["battery_levels"][0]["level"], 100.0)
        self.assertEqual(metrics["battery_levels"][2]["level"], 90.0)

    def test_final_metrics_calculation(self):
        """Test final metrics calculation."""
        # Add some test data
        for i in range(10):
            task_data = {
                "completed": i < 8,  # 8 completed, 2 missed
                "energy_used_wh": 0.001 if i < 8 else 0,
                "clean_energy_used_wh": 0.0008 if i < 8 else 0,
                "model_used": "YOLOv10_N" if i < 5 else "YOLOv10_X",
            }
            self.collector.update_task_metrics(task_data)

        self.collector.calculate_final_metrics()
        metrics = self.collector.get_metrics()

        # Check calculated metrics
        self.assertEqual(metrics["total_tasks"], 10)
        self.assertEqual(metrics["completed_tasks"], 8)
        self.assertEqual(metrics["task_completion_rate"], 80.0)
        self.assertEqual(metrics["total_energy_wh"], 0.008)
        self.assertAlmostEqual(metrics["clean_energy_wh"], 0.0064, places=6)
        self.assertAlmostEqual(metrics["clean_energy_percentage"], 80.0, places=6)

    def test_model_categorization(self):
        """Test model categorization into small/large."""
        # Test small models
        small_models = ["YOLOv10_N", "YOLOv10_S"]
        for model in small_models:
            self.collector.reset_metrics()
            task_data = {"completed": True, "model_used": model}
            self.collector.update_task_metrics(task_data)
            metrics = self.collector.get_metrics()
            self.assertEqual(metrics["small_model_tasks"], 1)
            self.assertEqual(metrics["large_model_tasks"], 0)

        # Test large models
        large_models = ["YOLOv10_M", "YOLOv10_B", "YOLOv10_L", "YOLOv10_X"]
        for model in large_models:
            self.collector.reset_metrics()
            task_data = {"completed": True, "model_used": model}
            self.collector.update_task_metrics(task_data)
            metrics = self.collector.get_metrics()
            self.assertEqual(metrics["small_model_tasks"], 0)
            self.assertEqual(metrics["large_model_tasks"], 1)

    def test_edge_cases(self):
        """Test edge cases for metrics calculation."""
        # Test with no tasks
        self.collector.calculate_final_metrics()
        metrics = self.collector.get_metrics()

        self.assertEqual(metrics["task_completion_rate"], 0.0)
        self.assertEqual(metrics["clean_energy_percentage"], 0.0)
        self.assertEqual(metrics["small_model_miss_rate"], 0.0)
        self.assertEqual(metrics["large_model_miss_rate"], 0.0)

        # Test with single task
        self.collector.reset_metrics()
        task_data = {
            "completed": True,
            "energy_used_wh": 0.001,
            "clean_energy_used_wh": 0.001,
            "model_used": "YOLOv10_N",
        }
        self.collector.update_task_metrics(task_data)
        self.collector.calculate_final_metrics()
        metrics = self.collector.get_metrics()

        self.assertEqual(metrics["task_completion_rate"], 100.0)
        self.assertEqual(metrics["clean_energy_percentage"], 100.0)


class TestCSVExporter(unittest.TestCase):
    """Unit tests for CSVExporter class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.exporter = CSVExporter(self.temp_dir)

        # Sample metrics for testing
        self.sample_metrics = {
            "total_tasks": 100,
            "completed_tasks": 95,
            "missed_deadlines": 5,
            "task_completion_rate": 95.0,
            "small_model_tasks": 60,
            "large_model_tasks": 40,
            "small_model_misses": 2,
            "large_model_misses": 3,
            "small_model_miss_rate": 3.33,
            "large_model_miss_rate": 7.5,
            "total_energy_wh": 0.5,
            "clean_energy_wh": 0.3,
            "clean_energy_percentage": 60.0,
            "final_battery_level": 25.0,
            "avg_battery_level": 75.0,
            "model_selections": {
                "YOLOv10_N": 30,
                "YOLOv10_S": 30,
                "YOLOv10_M": 20,
                "YOLOv10_X": 20,
            },
            "battery_levels": [
                {"timestamp": 0.0, "level": 100.0},
                {"timestamp": 5.0, "level": 95.0},
            ],
        }

        self.sample_simulation_info = {
            "id": "test_simulation",
            "location": "CA",
            "season": "summer",
            "week": 1,
            "controller": "naive_weak",
        }

    def test_summary_export(self):
        """Test CSV summary export."""
        summary_file = self.exporter.export_summary(
            self.sample_metrics, self.sample_simulation_info
        )

        # Should return valid file path
        self.assertIsNotNone(summary_file)
        self.assertTrue(isinstance(summary_file, str))

        # Verify CSV content by creating and reading
        with open(summary_file, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            self.assertEqual(len(rows), 1)
            row = rows[0]

            self.assertEqual(row["simulation_id"], "test_simulation")
            self.assertEqual(row["location"], "CA")
            self.assertEqual(row["season"], "summer")
            self.assertEqual(row["week"], "1")
            self.assertEqual(row["controller"], "naive_weak")
            self.assertEqual(row["total_tasks"], "100")
            self.assertEqual(row["task_completion_rate"], "95.0")
            self.assertEqual(row["clean_energy_percentage"], "60.0")
            self.assertEqual(row["YOLOv10_N_count"], "30")

        # Clean up test file
        Path(summary_file).unlink()

    def test_timeseries_export(self):
        """Test CSV time-series export."""
        timeseries_file = self.exporter.export_detailed_timeseries(
            self.sample_metrics, self.sample_simulation_info
        )

        # Should return valid file path
        self.assertIsNotNone(timeseries_file)
        self.assertTrue(isinstance(timeseries_file, str))

        # Verify CSV content by creating and reading
        with open(timeseries_file, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            self.assertEqual(len(rows), 2)

            # Check first row
            self.assertEqual(rows[0]["timestamp"], "0.0")
            self.assertEqual(rows[0]["battery_level"], "100.0")
            self.assertEqual(rows[0]["simulation_id"], "test_simulation")
            self.assertEqual(rows[0]["location"], "CA")

            # Check second row
            self.assertEqual(rows[1]["timestamp"], "5.0")
            self.assertEqual(rows[1]["battery_level"], "95.0")

        # Clean up test file
        Path(timeseries_file).unlink()

    def test_aggregated_export(self):
        """Test aggregated results export."""
        # Create multiple simulation results
        all_simulations = [
            {**self.sample_metrics, "controller": "naive_weak"},
            {**self.sample_metrics, "controller": "naive_strong", "total_tasks": 110},
            {**self.sample_metrics, "controller": "custom", "total_tasks": 105},
        ]

        aggregated_file = self.exporter.export_aggregated_results(all_simulations)

        # Should return valid file path
        self.assertIsNotNone(aggregated_file)
        self.assertTrue(isinstance(aggregated_file, str))

        # Verify CSV content by creating and reading
        with open(aggregated_file, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["controller"], "naive_weak")
            self.assertEqual(rows[1]["controller"], "naive_strong")
            self.assertEqual(rows[2]["controller"], "custom")
            self.assertEqual(rows[1]["total_tasks"], "110")
            self.assertEqual(rows[2]["total_tasks"], "105")

        # Clean up test file
        Path(aggregated_file).unlink()

    def test_json_export(self):
        """Test JSON export functionality."""
        test_data = {"key": "value", "number": 42}
        json_file = self.exporter.export_json(test_data, "test.json")

        # Should return valid file path
        self.assertIsNotNone(json_file)
        self.assertTrue(isinstance(json_file, str))

        # Verify JSON content by creating and reading
        with open(json_file, "r") as f:
            loaded_data = json.load(f)

            self.assertEqual(loaded_data["key"], "value")
            self.assertEqual(loaded_data["number"], 42)

        # Clean up test file
        Path(json_file).unlink()

    def test_empty_simulation_handling(self):
        """Test handling of empty simulation data."""
        empty_metrics = {"total_tasks": 0, "completed_tasks": 0, "battery_levels": []}

        # Should not raise errors
        summary_file = self.exporter.export_summary(
            empty_metrics, self.sample_simulation_info
        )
        timeseries_file = self.exporter.export_detailed_timeseries(
            empty_metrics, self.sample_simulation_info
        )

        # Summary should still be created (returns path)
        self.assertIsNotNone(summary_file)
        self.assertTrue(isinstance(summary_file, str))

        # Time-series should return empty string for no data
        self.assertEqual(timeseries_file, "")

    def test_data_integrity(self):
        """Test data integrity in exported files."""
        summary_file = self.exporter.export_summary(
            self.sample_metrics, self.sample_simulation_info
        )

        # Read back and verify data integrity
        with open(summary_file, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            row = rows[0]

            # Check numeric values are preserved as strings
            self.assertEqual(float(row["task_completion_rate"]), 95.0)
            self.assertEqual(float(row["clean_energy_percentage"]), 60.0)
            self.assertEqual(int(row["total_tasks"]), 100)

            # Check model counts
            self.assertEqual(int(row["YOLOv10_N_count"]), 30)
            self.assertEqual(int(row["YOLOv10_X_count"]), 20)


if __name__ == "__main__":
    unittest.main()
