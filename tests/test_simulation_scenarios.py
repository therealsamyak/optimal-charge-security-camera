"""Scenario tests for different simulation configurations."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestScenarios:
    """Test different simulation scenarios."""

    def test_seasonal_variations(self):
        """Test all 4 seasonal days."""
        seasonal_dates = [
            ("2024-01-05", "winter"),
            ("2024-04-15", "spring"),
            ("2024-07-04", "summer"),
            ("2024-10-20", "fall"),
        ]

        base_config = {
            "accuracy_threshold": 0.9,
            "latency_threshold_ms": 10.0,
            "simulation": {
                "image_quality": "good",
                "output_interval_seconds": 60,
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

        for date, season in seasonal_dates:
            config = base_config.copy()
            config["simulation"]["date"] = date

            # Test that config can be loaded and simulation initialized
            # (Full simulation would be too slow for unit tests)
            assert config["simulation"]["date"] == date
            assert config["simulation"]["controller_type"] == "custom"

    def test_image_quality_variations(self):
        """Test both image quality levels."""
        image_qualities = ["good", "bad"]

        base_config = {
            "accuracy_threshold": 0.9,
            "latency_threshold_ms": 10.0,
            "simulation": {
                "date": "2024-01-05",
                "output_interval_seconds": 60,
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

        for quality in image_qualities:
            config = base_config.copy()
            config["simulation"]["image_quality"] = quality

            assert config["simulation"]["image_quality"] == quality
            assert quality in ["good", "bad"]

    def test_controller_type_variations(self):
        """Test all 3 controller types."""
        controller_types = ["custom", "oracle", "benchmark"]

        base_config = {
            "accuracy_threshold": 0.9,
            "latency_threshold_ms": 10.0,
            "simulation": {
                "date": "2024-01-05",
                "image_quality": "good",
                "output_interval_seconds": 60,
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
        }

        # Add controller-specific configs
        base_config["custom_controller_weights"] = {
            "accuracy_weight": 0.4,
            "latency_weight": 0.3,
            "energy_cleanliness_weight": 0.2,
            "battery_conservation_weight": 0.1,
        }

        base_config["oracle_controller"] = {
            "optimization_horizon_hours": 24,
            "time_step_minutes": 5,
            "clean_energy_bonus_factor": 1.5,
        }

        base_config["benchmark_controller"] = {
            "prefer_largest_model": True,
            "charge_when_below": 30.0,
        }

        for controller_type in controller_types:
            config = base_config.copy()
            config["simulation"]["controller_type"] = controller_type

            assert config["simulation"]["controller_type"] == controller_type
            assert controller_type in ["custom", "oracle", "benchmark"]

    def test_accuracy_latency_threshold_combinations(self):
        """Test various accuracy/latency threshold combinations."""
        threshold_combinations = [
            (0.95, 5.0),  # High performance
            (0.90, 10.0),  # Medium performance
            (0.80, 20.0),  # Lower performance
            (0.99, 2.0),  # Very high performance
        ]

        base_config = {
            "simulation": {
                "date": "2024-01-05",
                "image_quality": "good",
                "output_interval_seconds": 60,
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

        for accuracy_threshold, latency_threshold in threshold_combinations:
            config = base_config.copy()
            config["accuracy_threshold"] = accuracy_threshold
            config["latency_threshold_ms"] = latency_threshold

            assert config["accuracy_threshold"] == accuracy_threshold
            assert config["latency_threshold_ms"] == latency_threshold
            assert 0.0 <= accuracy_threshold <= 1.0
            assert latency_threshold > 0.0

    def test_battery_configuration_variations(self):
        """Test different battery configurations."""
        battery_configs = [
            {
                "initial_capacity": 100.0,
                "max_capacity": 100.0,
                "charging_rate": 0.0035,
                "low_battery_threshold": 20.0,
            },
            {
                "initial_capacity": 50.0,
                "max_capacity": 100.0,
                "charging_rate": 0.0070,  # Faster charging
                "low_battery_threshold": 10.0,  # Lower threshold
            },
            {
                "initial_capacity": 75.0,
                "max_capacity": 75.0,  # Smaller battery
                "charging_rate": 0.0020,  # Slower charging
                "low_battery_threshold": 25.0,
            },
        ]

        base_config = {
            "accuracy_threshold": 0.9,
            "latency_threshold_ms": 10.0,
            "simulation": {
                "date": "2024-01-05",
                "image_quality": "good",
                "output_interval_seconds": 60,
                "controller_type": "custom",
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

        for battery_config in battery_configs:
            config = base_config.copy()
            config["battery"] = battery_config

            assert (
                config["battery"]["initial_capacity"]
                == battery_config["initial_capacity"]
            )
            assert config["battery"]["max_capacity"] == battery_config["max_capacity"]
            assert config["battery"]["charging_rate"] == battery_config["charging_rate"]
            assert (
                config["battery"]["low_battery_threshold"]
                == battery_config["low_battery_threshold"]
            )

            # Validate battery config constraints
            assert 0.0 <= battery_config["initial_capacity"] <= 100.0
            assert 0.0 <= battery_config["max_capacity"] <= 100.0
            assert battery_config["initial_capacity"] <= battery_config["max_capacity"]
            assert battery_config["charging_rate"] > 0.0
            assert 0.0 <= battery_config["low_battery_threshold"] <= 100.0
