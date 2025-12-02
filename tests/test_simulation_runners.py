#!/usr/bin/env python3
"""
Tests for simulation runner functionality.

Tests both basic and batch simulation runners without running full simulations.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config_loader import BatchConfig, ConfigLoader
from src.simulation_runner_base import SimulationRunnerBase


class TestConfigLoader(unittest.TestCase):
    """Test configuration loading and validation."""

    def setUp(self):
        """Set up test configuration."""
        self.test_config = {
            "simulation": {
                "duration_days": 7,
                "task_interval_seconds": 5,
                "time_acceleration": 1,
                "user_accuracy_requirement": 45.0,
                "user_latency_requirement": 8.0,
            },
            "battery": {"capacity_wh": 5.0, "charge_rate_watts": 100},
            "locations": ["CA", "FL", "NW", "NY"],
            "seasons": ["winter", "spring", "summer", "fall"],
            "controllers": ["naive_weak", "naive_strong", "oracle", "custom"],
            "output_dir": "results/",
            "batch_run": {
                "num_variations": 5,
                "random_seed": 42,
                "output_detailed_csv": True,
                "accuracy_range": {"min": 30, "max": 80},
                "latency_range": {"min": 2, "max": 15},
                "battery_capacity_range": {"min": 2, "max": 15},
                "charge_rate_range": {"min": 50, "max": 200},
            },
        }

    def test_batch_config_creation(self):
        """Test BatchConfig object creation."""
        batch_config = BatchConfig(
            num_variations=5,
            random_seed=42,
            output_detailed_csv=True,
            accuracy_range={"min": 30, "max": 80},
            latency_range={"min": 2, "max": 15},
            battery_capacity_range={"min": 2, "max": 15},
            charge_rate_range={"min": 50, "max": 200},
        )

        self.assertEqual(batch_config.num_variations, 5)
        self.assertEqual(batch_config.random_seed, 42)
        self.assertTrue(batch_config.output_detailed_csv)
        self.assertEqual(batch_config.accuracy_range["min"], 30)
        self.assertEqual(batch_config.accuracy_range["max"], 80)

    def test_config_loader_batch_config(self):
        """Test loading batch configuration."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(self.test_config, f)
            config_path = f.name

        try:
            config_loader = ConfigLoader(config_path)
            batch_config = config_loader.get_batch_config()

            self.assertEqual(batch_config.num_variations, 5)
            self.assertEqual(batch_config.random_seed, 42)
            self.assertTrue(batch_config.output_detailed_csv)
            self.assertEqual(batch_config.accuracy_range["min"], 30)
            self.assertEqual(batch_config.accuracy_range["max"], 80)

        finally:
            Path(config_path).unlink()

    def test_config_validation_with_batch(self):
        """Test configuration validation including batch section."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(self.test_config, f)
            config_path = f.name

        try:
            config_loader = ConfigLoader(config_path)
            self.assertTrue(config_loader.validate_config())

        finally:
            Path(config_path).unlink()

    def test_config_validation_invalid_batch_ranges(self):
        """Test configuration validation with invalid batch ranges."""
        invalid_config = self.test_config.copy()
        invalid_config["batch_run"]["accuracy_range"]["min"] = 80
        invalid_config["batch_run"]["accuracy_range"]["max"] = 30  # min > max

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(invalid_config, f)
            config_path = f.name

        try:
            config_loader = ConfigLoader(config_path)
            self.assertFalse(config_loader.validate_config())

        finally:
            Path(config_path).unlink()


class TestSimulationRunnerBase(unittest.TestCase):
    """Test simulation runner base functionality."""

    def setUp(self):
        """Set up test environment."""
        self.test_config = {
            "simulation": {
                "duration_days": 1,  # Short duration for testing
                "task_interval_seconds": 1,
                "time_acceleration": 1,
                "user_accuracy_requirement": 45.0,
                "user_latency_requirement": 8.0,
            },
            "battery": {"capacity_wh": 5.0, "charge_rate_watts": 100},
            "locations": ["CA", "FL"],  # Fewer locations for testing
            "seasons": ["winter", "spring"],  # Fewer seasons for testing
            "controllers": [
                "naive_weak",
                "naive_strong",
            ],  # Fewer controllers for testing
            "output_dir": "test_results/",
            "batch_run": {
                "num_variations": 2,
                "random_seed": 42,
                "output_detailed_csv": True,
                "accuracy_range": {"min": 30, "max": 80},
                "latency_range": {"min": 2, "max": 15},
                "battery_capacity_range": {"min": 2, "max": 15},
                "charge_rate_range": {"min": 50, "max": 200},
            },
        }

        # Create temporary config file
        self.temp_config_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        )
        json.dump(self.test_config, self.temp_config_file)
        self.temp_config_file.close()

        # Create temporary results directory
        self.temp_results_dir = tempfile.mkdtemp()

        # Mock power profiles
        self.mock_power_profiles = {"CA": {"winter": {"week_1": []}}}

    def tearDown(self):
        """Clean up test environment."""
        Path(self.temp_config_file.name).unlink()
        import shutil

        shutil.rmtree(self.temp_results_dir, ignore_errors=True)

    @patch("simulation_runner_base.SimulationRunnerBase._load_power_profiles")
    def test_simulation_list_generation(self, mock_load_profiles):
        """Test simulation list generation."""
        mock_load_profiles.return_value = self.mock_power_profiles

        with patch("sys.path"):
            runner = SimulationRunnerBase(self.temp_config_file.name)

            # Test basic simulation list (week 1 only)
            simulations = runner._generate_simulation_list(
                locations=["CA", "FL"],
                seasons=["winter", "spring"],
                controllers=["naive_weak", "naive_strong"],
                weeks=[1],
            )

            # Should generate 2 locations × 2 seasons × 2 controllers × 1 week = 8 simulations
            self.assertEqual(len(simulations), 8)

            # Check structure of simulation parameters
            sim = simulations[0]
            self.assertIn("location", sim)
            self.assertIn("season", sim)
            self.assertIn("week", sim)
            self.assertIn("controller", sim)
            self.assertIn("simulation_id", sim)

    @patch("simulation_runner_base.SimulationRunnerBase._load_power_profiles")
    def test_parameter_variations_generation(self, mock_load_profiles):
        """Test parameter variations generation."""
        mock_load_profiles.return_value = self.mock_power_profiles

        with patch("sys.path"):
            runner = SimulationRunnerBase(self.temp_config_file.name)
            batch_config = BatchConfig(
                num_variations=3,
                random_seed=42,
                accuracy_range={"min": 30, "max": 80},
                latency_range={"min": 2, "max": 15},
                battery_capacity_range={"min": 2, "max": 15},
                charge_rate_range={"min": 50, "max": 200},
            )

            variations = runner.generate_parameter_variations(batch_config)

            self.assertEqual(len(variations), 3)

            # Check variation structure
            for variation in variations:
                self.assertIn("variation_id", variation)
                self.assertIn("accuracy_requirement", variation)
                self.assertIn("latency_requirement", variation)
                self.assertIn("battery_capacity_wh", variation)
                self.assertIn("charge_rate_watts", variation)

                # Check parameter ranges
                self.assertGreaterEqual(variation["accuracy_requirement"], 30)
                self.assertLessEqual(variation["accuracy_requirement"], 80)
                self.assertGreaterEqual(variation["latency_requirement"], 2)
                self.assertLessEqual(variation["latency_requirement"], 15)

    @patch("simulation_runner_base.SimulationRunnerBase._load_power_profiles")
    def test_simulation_config_with_overrides(self, mock_load_profiles):
        """Test simulation config creation with parameter overrides."""
        mock_load_profiles.return_value = self.mock_power_profiles

        with patch("sys.path"):
            runner = SimulationRunnerBase(self.temp_config_file.name)
            base_config = runner.config_loader.get_simulation_config()

            # Test with overrides
            overridden_config = runner._create_simulation_config_with_overrides(
                base_config=base_config,
                accuracy_override=60.0,
                latency_override=10.0,
                battery_capacity_override=8.0,
                charge_rate_override=150.0,
            )

            self.assertEqual(overridden_config.user_accuracy_requirement, 60.0)
            self.assertEqual(overridden_config.user_latency_requirement, 10.0)
            self.assertEqual(overridden_config.battery_capacity_wh, 8.0)
            self.assertEqual(overridden_config.charge_rate_watts, 150.0)

            # Test without overrides (should use base config values)
            default_config = runner._create_simulation_config_with_overrides(
                base_config=base_config
            )
            self.assertEqual(
                default_config.user_accuracy_requirement,
                base_config.user_accuracy_requirement,
            )
            self.assertEqual(
                default_config.user_latency_requirement,
                base_config.user_latency_requirement,
            )


class TestCSVExporter(unittest.TestCase):
    """Test CSV export functionality."""

    def setUp(self):
        """Set up test environment."""
        from src.metrics_collector import CSVExporter

        self.temp_dir = tempfile.mkdtemp()
        self.exporter = CSVExporter(self.temp_dir)

        # Sample simulation data
        self.sample_simulations = [
            {
                "simulation_id": "CA_winter_week1_naive_weak",
                "location": "CA",
                "season": "winter",
                "week": 1,
                "controller": "naive_weak",
                "total_tasks": 100,
                "completed_tasks": 95,
                "task_completion_rate": 95.0,
                "total_energy_wh": 50.5,
                "clean_energy_wh": 30.2,
                "clean_energy_percentage": 59.8,
                "success": True,
                "timestamp": "2025-11-30T16:24:42.785",
            },
            {
                "simulation_id": "CA_winter_week1_naive_strong",
                "location": "CA",
                "season": "winter",
                "week": 1,
                "controller": "naive_strong",
                "total_tasks": 100,
                "completed_tasks": 98,
                "task_completion_rate": 98.0,
                "total_energy_wh": 75.3,
                "clean_energy_wh": 45.1,
                "clean_energy_percentage": 59.9,
                "success": True,
                "timestamp": "2025-11-30T16:24:42.785",
            },
        ]

    def tearDown(self):
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_export_aggregated_results(self):
        """Test exporting aggregated results."""
        filename = "test_aggregated.csv"
        filepath = self.exporter.export_aggregated_results(
            self.sample_simulations, filename
        )

        self.assertTrue(Path(filepath).exists())
        self.assertTrue(filepath.endswith(filename))

        # Check file contents
        with open(filepath, "r") as f:
            content = f.read()
            self.assertIn("simulation_id", content)
            self.assertIn("CA_winter_week1_naive_weak", content)
            self.assertIn("CA_winter_week1_naive_strong", content)

    def test_export_detailed_results(self):
        """Test exporting detailed results with parameters."""
        # Add parameter variations to sample data
        detailed_simulations = []
        for i, sim in enumerate(self.sample_simulations):
            detailed_sim = sim.copy()
            detailed_sim.update(
                {
                    "variation_id": i + 1,
                    "accuracy_requirement": 45.0 + i * 5,
                    "latency_requirement": 8.0 + i * 2,
                    "battery_capacity_wh": 5.0 + i,
                    "charge_rate_watts": 100.0 + i * 50,
                }
            )
            detailed_simulations.append(detailed_sim)

        filename = "test_detailed.csv"
        filepath = self.exporter.export_detailed_results(detailed_simulations, filename)

        self.assertTrue(Path(filepath).exists())
        self.assertTrue(filepath.endswith(filename))

        # Check file contents
        with open(filepath, "r") as f:
            content = f.read()
            self.assertIn("variation_id", content)
            self.assertIn("accuracy_requirement", content)
            self.assertIn("latency_requirement", content)
            self.assertIn("battery_capacity_wh", content)
            self.assertIn("charge_rate_watts", content)


if __name__ == "__main__":
    unittest.main()
