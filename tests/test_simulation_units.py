"""Unit tests for simulation components."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.simulation.controllers import (
    CustomController,
    OracleController,
    BenchmarkController,
)
from src.sensors.simulation_sensors import SimulationSensors
from src.simulation.metrics import MetricsTracker


class TestControllers:
    """Test controller decision logic."""

    def test_custom_controller_selection(self):
        """Test custom controller weighted scoring."""
        config = {
            "accuracy_threshold": 0.9,
            "latency_threshold_ms": 10.0,
            "custom_controller_weights": {
                "accuracy_weight": 0.4,
                "latency_weight": 0.3,
                "energy_cleanliness_weight": 0.2,
                "battery_conservation_weight": 0.1,
            },
        }

        controller = CustomController(config)

        model_data = {
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

        # Should select YOLOv10-S (meets thresholds, good balance)
        selected = controller.select_model(80.0, 0.7, model_data)
        assert selected == "YOLOv10-S"

    def test_oracle_controller_selection(self):
        """Test oracle controller efficiency preference."""
        config = {
            "accuracy_threshold": 0.9,
            "latency_threshold_ms": 10.0,
            "oracle_controller": {
                "optimization_horizon_hours": 24,
                "time_step_minutes": 5,
                "clean_energy_bonus_factor": 1.5,
            },
        }

        controller = OracleController(config)

        model_data = {
            "YOLOv10-N": {
                "accuracy": 0.95,
                "latency_ms": 5.0,
                "energy_consumption": 0.004,
            },
            "YOLOv10-S": {
                "accuracy": 0.92,
                "latency_ms": 8.0,
                "energy_consumption": 0.007,
            },
        }

        # Should select most efficient model (YOLOv10-N)
        selected = controller.select_model(80.0, 0.7, model_data)
        assert selected == "YOLOv10-N"

    def test_benchmark_controller_selection(self):
        """Test benchmark controller prefers largest model."""
        config = {
            "accuracy_threshold": 0.9,
            "latency_threshold_ms": 10.0,
            "benchmark_controller": {
                "prefer_largest_model": True,
                "charge_when_below": 30.0,
            },
        }

        controller = BenchmarkController(config)

        model_data = {
            "YOLOv10-N": {
                "accuracy": 0.85,
                "latency_ms": 5.0,
                "energy_consumption": 0.004,
            },
            "YOLOv10-X": {
                "accuracy": 0.98,
                "latency_ms": 15.0,
                "energy_consumption": 0.023,
            },
        }

        # Should select largest available model
        selected = controller.select_model(80.0, 0.7, model_data)
        assert selected == "YOLOv10-X"

        # Should force charging when battery low
        selected = controller.select_model(25.0, 0.7, model_data)
        assert selected is None


class TestSimulationSensors:
    """Test sensor functionality."""

    def test_battery_consumption(self):
        """Test battery energy consumption."""
        config = {
            "battery": {
                "initial_capacity": 100.0,
                "max_capacity": 100.0,
                "charging_rate": 0.0035,
                "low_battery_threshold": 20.0,
            }
        }

        sensors = SimulationSensors(config)

        # Test energy consumption
        assert sensors.consume_energy(5.0)
        assert sensors.get_battery_level() == 95.0

        # Test insufficient energy
        assert not sensors.consume_energy(200.0)
        assert sensors.get_battery_level() == 95.0

    def test_battery_charging(self):
        """Test battery charging."""
        config = {
            "battery": {
                "initial_capacity": 50.0,
                "max_capacity": 100.0,
                "charging_rate": 0.0035,
                "low_battery_threshold": 20.0,
            }
        }

        sensors = SimulationSensors(config)

        # Test charging
        sensors.charge_battery(1000)  # 1000 seconds
        expected_charge = 50.0 + (0.0035 * 1000)
        assert sensors.get_battery_level() == min(100.0, expected_charge)

    def test_charging_detection(self):
        """Test charging threshold detection."""
        config = {
            "battery": {
                "initial_capacity": 15.0,
                "max_capacity": 100.0,
                "charging_rate": 0.0035,
                "low_battery_threshold": 20.0,
            }
        }

        sensors = SimulationSensors(config)
        assert sensors.is_charging()

        sensors.current_capacity = 25.0
        assert not sensors.is_charging()


class TestMetricsTracker:
    """Test metrics calculation."""

    def test_miss_tracking(self):
        """Test miss type tracking."""
        tracker = MetricsTracker()

        # Add various result types
        tracker.add_result(
            {
                "miss_type": "none",
                "model_selected": "YOLOv10-S",
                "energy_consumed": 0.007,
                "clean_energy_consumed": 0.005,
            }
        )

        tracker.add_result(
            {
                "miss_type": "small_miss",
                "model_selected": "YOLOv10-N",
                "energy_consumed": 0.004,
                "clean_energy_consumed": 0.002,
            }
        )

        tracker.add_result(
            {
                "miss_type": "large_miss",
                "model_selected": "",
                "energy_consumed": 0.0,
                "clean_energy_consumed": 0.0,
            }
        )

        summary = tracker.get_summary()

        assert summary["total_inferences"] == 2  # Only actual inferences count
        assert summary["small_misses"] == 1
        assert summary["large_misses"] == 1
        assert summary["small_miss_rate"] == 50.0
        assert summary["large_miss_rate"] == 50.0

    def test_energy_tracking(self):
        """Test energy consumption tracking."""
        tracker = MetricsTracker()

        tracker.add_result(
            {
                "miss_type": "none",
                "model_selected": "YOLOv10-S",
                "energy_consumed": 0.007,
                "clean_energy_consumed": 0.005,
            }
        )

        tracker.add_result(
            {
                "miss_type": "none",
                "model_selected": "YOLOv10-N",
                "energy_consumed": 0.004,
                "clean_energy_consumed": 0.002,
            }
        )

        summary = tracker.get_summary()

        assert summary["total_energy_used"] == 0.011
        assert summary["clean_energy_used"] == 0.007
        assert summary["clean_energy_percentage"] == (0.007 / 0.011) * 100
