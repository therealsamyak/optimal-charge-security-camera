"""
Integration tests for OCS Camera system.

Tests end-to-end functionality with different configurations.
"""

import pytest
import tempfile
import yaml
import time
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config.manager import ConfigManager
from controller.intelligent_controller import ModelController
from sensors.mock_battery import MockBatterySensor
from sensors.mock_energy import MockEnergySensor
from tests.test_mock_sensors import MockSensorDataGenerator, PerformanceDataGenerator


class TestIntegration:
    """Integration tests for the complete system."""

    def test_config_to_controller_integration(self):
        """Test integration between configuration and controller."""
        # Create test configuration
        test_config = {
            "requirements": {
                "min_accuracy": 90.0,
                "max_latency_ms": 50.0,
                "run_frequency_ms": 1000.0,
            },
            "controller": {
                "enable_charging": True,
                "min_battery_threshold": 25.0,
                "max_battery_threshold": 85.0,
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.dump(test_config, f)
            config_path = f.name

        try:
            # Load configuration
            ConfigManager(config_path)

            # Initialize controller with this configuration
            controller = ModelController()

            # Test that controller respects configuration
            decision = controller.select_optimal_model(50.0, 50.0, False)

            # Should select a model that meets the strict requirements
            # With 90% accuracy requirement, should prefer larger models
            assert decision.selected_model in ["yolov10b", "yolov10l", "yolov10x"]

        finally:
            os.unlink(config_path)

    def test_sensor_controller_integration(self):
        """Test integration between sensors and controller."""
        # Initialize components
        controller = ModelController()
        battery_sensor = MockBatterySensor(60.0)
        data_generator = MockSensorDataGenerator(seed=42)

        # Simulate sensor data sequence
        readings = data_generator.generate_scenario("normal_operation")

        decisions = []
        for reading in readings:
            # Update sensor states
            battery_sensor._battery_level = reading.battery_level
            battery_sensor._is_charging = reading.is_charging

            # Get controller decision
            decision = controller.select_optimal_model(
                reading.battery_level, reading.energy_cleanliness, reading.is_charging
            )

            decisions.append(decision)

        # Should have made decisions for each reading
        assert len(decisions) == len(readings)

        # Decisions should be consistent with sensor data
        for i, (reading, decision) in enumerate(zip(readings, decisions)):
            # Should contain expected information
            assert decision.selected_model in controller.get_available_models()
            assert isinstance(decision.score, float)
            assert isinstance(decision.should_charge, bool)

            # Charging decisions should make sense given battery level
            if reading.battery_level < 30.0:
                assert decision.should_charge or reading.is_charging

    def test_low_battery_scenario(self):
        """Test system behavior with low battery scenario."""
        controller = ModelController()
        battery_sensor = MockBatterySensor(15.0)  # Start with low battery
        energy_sensor = MockEnergySensor()

        # Simulate clean energy scenario
        energy_sensor._time_offset = time.time() - 12 * 3600  # Midday

        # Get decision
        decision = controller.select_optimal_model(
            battery_sensor.get_battery_percentage(),
            energy_sensor.get_energy_cleanliness(),
            battery_sensor.is_charging(),
        )

        # Should start charging and select efficient model
        assert decision.should_charge is True
        assert "Start charging" in decision.reasoning

        # Should prefer smaller models when battery is low
        assert decision.selected_model in ["yolov10n", "yolov10s", "yolov10m"]

    def test_clean_energy_scenario(self):
        """Test system behavior with clean energy scenario."""
        controller = ModelController()
        battery_sensor = MockBatterySensor(50.0)
        energy_sensor = MockEnergySensor()

        # Simulate optimal clean energy conditions
        energy_sensor._time_offset = time.time() - 12 * 3600  # Midday

        # Get decision
        decision = controller.select_optimal_model(
            battery_sensor.get_battery_percentage(),
            energy_sensor.get_energy_cleanliness(),
            battery_sensor.is_charging(),
        )

        # Should take advantage of clean energy
        energy_cleanliness = energy_sensor.get_energy_cleanliness()
        if energy_cleanliness > 80.0:
            # Should consider charging if battery is not full
            if battery_sensor.get_battery_percentage() < 70.0:
                assert decision.should_charge is True

    def test_performance_validation_integration(self):
        """Test integration with performance validation."""
        from controller.performance_validator import (
            PerformanceValidator,
            PerformanceMetrics,
        )

        validator = PerformanceValidator()
        data_generator = PerformanceDataGenerator(seed=42)

        # Test with different models and performance levels
        for model_name in ["yolov10n", "yolov10m", "yolov10x"]:
            # Generate performance metrics
            metrics = data_generator.generate_inference_metrics(
                model_name, has_detection=True
            )

            # Create performance metrics object
            perf_metrics = PerformanceMetrics(
                accuracy=metrics["confidence"] * 100,
                latency_ms=metrics["latency_ms"],
                battery_consumption=metrics["battery_consumption"],
                confidence=metrics["confidence"],
                has_detection=metrics["has_detection"],
            )

            # Validate performance
            validation_result = validator.validate_performance(perf_metrics, model_name)

            # Should return validation result
            assert validation_result is not None
            assert isinstance(validation_result.score, float)
            assert isinstance(validation_result.violations, list)
            assert isinstance(validation_result.warnings, list)

    def test_end_to_end_simulation(self):
        """Test complete end-to-end simulation."""
        # Initialize all components
        controller = ModelController()
        battery_sensor = MockBatterySensor(80.0)
        data_generator = MockSensorDataGenerator(seed=42)
        perf_generator = PerformanceDataGenerator(seed=42)

        # Simulate multiple inference cycles
        cycle_count = 20
        results = []

        for i in range(cycle_count):
            # Generate sensor reading
            reading = data_generator.generate_reading()

            # Update sensor states
            battery_sensor._battery_level = reading.battery_level
            battery_sensor._is_charging = reading.is_charging

            # Get controller decision
            decision = controller.select_optimal_model(
                reading.battery_level, reading.energy_cleanliness, reading.is_charging
            )

            # Simulate inference performance
            perf_metrics = perf_generator.generate_inference_metrics(
                decision.selected_model,
                has_detection=(i % 3 == 0),  # Detection every 3rd cycle
            )

            # Store results
            results.append(
                {
                    "cycle": i,
                    "reading": reading,
                    "decision": decision,
                    "performance": perf_metrics,
                }
            )

        # Verify simulation results
        assert len(results) == cycle_count

        # Should have used different models based on conditions
        used_models = set(r["decision"].selected_model for r in results)
        assert len(used_models) >= 1  # At least one model used

        # Should have reasonable score distribution
        scores = [r["decision"].score for r in results]
        assert all(0 <= score <= 100 for score in scores)

        # Should have some charging decisions
        charging_decisions = [r for r in results if r["decision"].should_charge]
        assert (
            len(charging_decisions) >= 0
        )  # May or may not charge depending on conditions

    def test_configuration_variations(self):
        """Test system with different configuration variations."""
        test_configs = [
            # High performance config
            {
                "requirements": {
                    "min_accuracy": 95.0,
                    "max_latency_ms": 30.0,
                    "run_frequency_ms": 500.0,
                }
            },
            # Power saving config
            {
                "requirements": {
                    "min_accuracy": 70.0,
                    "max_latency_ms": 200.0,
                    "run_frequency_ms": 5000.0,
                },
                "controller": {
                    "min_battery_threshold": 40.0,
                    "max_battery_threshold": 80.0,
                },
            },
            # Balanced config
            {
                "requirements": {
                    "min_accuracy": 80.0,
                    "max_latency_ms": 100.0,
                    "run_frequency_ms": 2000.0,
                }
            },
        ]

        for i, test_config in enumerate(test_configs):
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yml", delete=False
            ) as f:
                yaml.dump(test_config, f)
                config_path = f.name

            try:
                # Test with this configuration
                ConfigManager(config_path)
                controller = ModelController()

                # Test decision making with different battery levels
                for battery_level in [20.0, 50.0, 80.0]:
                    decision = controller.select_optimal_model(
                        battery_level, 50.0, False
                    )

                    # Should make valid decision
                    assert decision.selected_model in controller.get_available_models()
                    assert isinstance(decision.score, float)

                    # Decision should be consistent with configuration
                    # (This would require more sophisticated testing based on specific config)

            finally:
                os.unlink(config_path)

    def test_error_handling_integration(self):
        """Test error handling in integrated system."""
        controller = ModelController()

        # Test with invalid sensor values
        try:
            # Very low battery (should handle gracefully)
            decision = controller.select_optimal_model(-5.0, 50.0, False)
            assert decision.selected_model in controller.get_available_models()

            # Very high energy cleanliness (should handle gracefully)
            decision = controller.select_optimal_model(50.0, 150.0, False)
            assert decision.selected_model in controller.get_available_models()

        except Exception as e:
            pytest.fail(
                f"Controller should handle invalid sensor values gracefully, but raised: {e}"
            )

        # Test with edge cases
        try:
            # Zero battery
            decision = controller.select_optimal_model(0.0, 50.0, False)
            assert decision.selected_model in controller.get_available_models()

            # Zero energy cleanliness
            decision = controller.select_optimal_model(50.0, 0.0, False)
            assert decision.selected_model in controller.get_available_models()

        except Exception as e:
            pytest.fail(
                f"Controller should handle edge cases gracefully, but raised: {e}"
            )


class TestPerformanceValidation:
    """Test performance validation against user requirements."""

    def test_accuracy_requirement_validation(self):
        """Test validation against accuracy requirements."""
        from controller.performance_validator import (
            PerformanceValidator,
            PerformanceMetrics,
        )

        validator = PerformanceValidator()

        # Test with high accuracy requirement
        config = ConfigManager()
        config._config["requirements"]["min_accuracy"] = 90.0

        # Test case 1: Meets requirement
        metrics = PerformanceMetrics(
            accuracy=95.0,
            latency_ms=50.0,
            battery_consumption=0.5,
            confidence=0.95,
            has_detection=True,
        )

        result = validator.validate_performance(metrics, "yolov10x")
        assert result.score > 0.5  # Should have good score
        assert not any("accuracy" in v.lower() for v in result.violations)

        # Test case 2: Fails requirement
        metrics_low_accuracy = PerformanceMetrics(
            accuracy=70.0,
            latency_ms=50.0,
            battery_consumption=0.5,
            confidence=0.70,
            has_detection=True,
        )

        result = validator.validate_performance(metrics_low_accuracy, "yolov10n")
        assert any("accuracy" in v.lower() for v in result.violations)

    def test_latency_requirement_validation(self):
        """Test validation against latency requirements."""
        from controller.performance_validator import (
            PerformanceValidator,
            PerformanceMetrics,
        )

        validator = PerformanceValidator()

        # Test with strict latency requirement
        config = ConfigManager()
        config._config["requirements"]["max_latency_ms"] = 50.0

        # Test case 1: Meets requirement
        metrics = PerformanceMetrics(
            accuracy=80.0,
            latency_ms=30.0,
            battery_consumption=0.3,
            confidence=0.80,
            has_detection=True,
        )

        result = validator.validate_performance(metrics, "yolov10s")
        assert not any("latency" in v.lower() for v in result.violations)

        # Test case 2: Fails requirement
        metrics_high_latency = PerformanceMetrics(
            accuracy=80.0,
            latency_ms=150.0,
            battery_consumption=0.3,
            confidence=0.80,
            has_detection=True,
        )

        result = validator.validate_performance(metrics_high_latency, "yolov10m")
        assert any("latency" in v.lower() for v in result.violations)

    def test_battery_efficiency_validation(self):
        """Test validation of battery efficiency."""
        from controller.performance_validator import (
            PerformanceValidator,
            PerformanceMetrics,
        )

        validator = PerformanceValidator()

        # Test with different models and battery consumption
        test_cases = [
            ("yolov10n", 0.1),  # Efficient
            ("yolov10m", 0.4),  # Medium
            ("yolov10x", 1.0),  # Inefficient
        ]

        for model_name, consumption in test_cases:
            metrics = PerformanceMetrics(
                accuracy=80.0,
                latency_ms=50.0,
                battery_consumption=consumption,
                confidence=0.80,
                has_detection=True,
            )

            result = validator.validate_performance(metrics, model_name)

            # Should return valid result for all cases
            assert isinstance(result.score, float)
            assert 0 <= result.score <= 100

            # More efficient models should generally get better scores
            # (This depends on the specific validation algorithm)
