"""
Mock sensor data generators for testing.

This module provides utilities to generate realistic mock sensor data
for testing the OCS Camera system.
"""

import random
import time
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class SensorReading:
    """Single sensor reading with timestamp."""

    timestamp: float
    battery_level: float
    energy_cleanliness: float
    is_charging: bool
    energy_source: str


class MockSensorDataGenerator:
    """Generates realistic mock sensor data for testing."""

    def __init__(self, seed: int | None = None):
        """
        Initialize sensor data generator.

        Args:
            seed: Random seed for reproducible data
        """
        if seed is not None:
            random.seed(seed)

        self.start_time = time.time()
        self.current_battery = 80.0
        self.is_charging = False
        self.energy_sources = ["solar", "wind", "hydro", "grid"]

    def generate_reading(self) -> SensorReading:
        """Generate a single sensor reading."""
        current_time = time.time() - self.start_time
        hour_of_day = (current_time / 3600) % 24

        # Simulate battery level changes
        if self.is_charging:
            # Charge at 2% per minute when charging
            self.current_battery = min(100.0, self.current_battery + 0.033)
        else:
            # Natural discharge at 0.1% per hour
            self.current_battery = max(0.0, self.current_battery - 0.000028)

        # Add small random fluctuation
        battery_level = max(
            0.0, min(100.0, self.current_battery + random.uniform(-0.5, 0.5))
        )

        # Simulate energy cleanliness based on time of day
        if 6 <= hour_of_day <= 18:  # Daylight hours
            base_cleanliness = 70.0 + random.uniform(-10, 20)
        else:  # Night hours
            base_cleanliness = 30.0 + random.uniform(-10, 15)

        energy_cleanliness = max(
            0.0, min(100.0, base_cleanliness + random.uniform(-5, 5))
        )

        # Determine energy source
        if 6 <= hour_of_day <= 18 and energy_cleanliness > 60:
            if random.random() < 0.7:
                energy_source = "solar"
            elif random.random() < 0.5:
                energy_source = "wind"
            else:
                energy_source = "hydro"
        else:
            energy_source = "grid"

        # Simulate charging decisions
        if battery_level < 30.0:
            self.is_charging = True
        elif battery_level > 90.0:
            self.is_charging = False
        elif energy_cleanliness > 85.0 and battery_level < 70.0:
            self.is_charging = True
        elif energy_cleanliness < 40.0 and battery_level > 60.0:
            self.is_charging = False

        return SensorReading(
            timestamp=time.time(),
            battery_level=battery_level,
            energy_cleanliness=energy_cleanliness,
            is_charging=self.is_charging,
            energy_source=energy_source,
        )

    def generate_sequence(
        self, count: int, interval_seconds: float = 1.0
    ) -> List[SensorReading]:
        """
        Generate a sequence of sensor readings.

        Args:
            count: Number of readings to generate
            interval_seconds: Time interval between readings

        Returns:
            List of sensor readings
        """
        readings = []
        for _ in range(count):
            readings.append(self.generate_reading())
            time.sleep(interval_seconds)
        return readings

    def generate_scenario(self, scenario: str) -> List[SensorReading]:
        """
        Generate sensor data for specific test scenarios.

        Args:
            scenario: Type of scenario ('low_battery', 'clean_energy', 'normal_operation')

        Returns:
            List of sensor readings for the scenario
        """
        if scenario == "low_battery":
            self.current_battery = 15.0
            return self.generate_sequence(50, 0.1)

        elif scenario == "clean_energy":
            # Set time to midday for maximum solar
            self.start_time = time.time() - 12 * 3600
            self.current_battery = 50.0
            return self.generate_sequence(50, 0.1)

        elif scenario == "normal_operation":
            self.current_battery = 80.0
            return self.generate_sequence(50, 0.1)

        else:
            raise ValueError(f"Unknown scenario: {scenario}")

    def get_battery_drain_profile(self, model_name: str) -> float:
        """
        Get battery drain rate for a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Battery drain rate per inference
        """
        drain_rates = {
            "yolov10n": 0.1,
            "yolov10s": 0.2,
            "yolov10m": 0.4,
            "yolov10b": 0.6,
            "yolov10l": 0.8,
            "yolov10x": 1.0,
        }
        return drain_rates.get(model_name, 0.5)

    def simulate_inference_cycle(
        self, model_name: str, count: int = 10
    ) -> List[SensorReading]:
        """
        Simulate battery drain during inference cycles.

        Args:
            model_name: Model being used for inference
            count: Number of inference cycles

        Returns:
            List of sensor readings during inference
        """
        readings = []
        drain_rate = self.get_battery_drain_profile(model_name)

        for _ in range(count):
            # Generate reading
            reading = self.generate_reading()

            # Apply battery drain from inference
            self.current_battery = max(0.0, self.current_battery - drain_rate)
            reading.battery_level = self.current_battery

            readings.append(reading)

            # Small delay between inferences
            time.sleep(0.05)

        return readings


class PerformanceDataGenerator:
    """Generates realistic performance data for testing."""

    def __init__(self, seed: int | None = None):
        """Initialize performance data generator."""
        if seed is not None:
            random.seed(seed)

    def generate_inference_metrics(
        self, model_name: str, has_detection: bool = False
    ) -> Dict:
        """
        Generate realistic inference metrics for a model.

        Args:
            model_name: Name of the model
            has_detection: Whether inference should show detection

        Returns:
            Dictionary of inference metrics
        """
        # Base performance characteristics by model
        model_specs = {
            "yolov10n": {"base_latency": 15, "latency_var": 5, "base_conf": 0.6},
            "yolov10s": {"base_latency": 25, "latency_var": 8, "base_conf": 0.7},
            "yolov10m": {"base_latency": 40, "latency_var": 12, "base_conf": 0.8},
            "yolov10b": {"base_latency": 60, "latency_var": 18, "base_conf": 0.85},
            "yolov10l": {"base_latency": 85, "latency_var": 25, "base_conf": 0.9},
            "yolov10x": {"base_latency": 120, "latency_var": 35, "base_conf": 0.94},
        }

        specs = model_specs.get(model_name, model_specs["yolov10m"])

        # Generate latency with some variation
        latency = max(
            5,
            specs["base_latency"]
            + random.uniform(-specs["latency_var"], specs["latency_var"]),
        )

        if has_detection:
            confidence = min(
                0.99, max(0.1, specs["base_conf"] + random.uniform(-0.1, 0.1))
            )
            label = random.choice(["person", "car", "bicycle", "dog", "chair"])
        else:
            confidence = random.uniform(
                0.0, 0.05
            )  # Very low confidence for no detection
            label = "no detection"

        return {
            "model_name": model_name,
            "latency_ms": latency,
            "confidence": confidence,
            "has_detection": has_detection,
            "label": label,
            "battery_consumption": specs["base_latency"] / 100.0,  # Rough estimate
            "memory_usage_mb": random.uniform(50, 200),
            "cpu_usage_percent": random.uniform(10, 40),
        }

    def generate_performance_sequence(
        self, model_name: str, count: int, detection_rate: float = 0.3
    ) -> List[Dict]:
        """
        Generate a sequence of performance metrics.

        Args:
            model_name: Model to generate metrics for
            count: Number of metrics to generate
            detection_rate: Probability of detection (0.0-1.0)

        Returns:
            List of performance metric dictionaries
        """
        metrics = []
        for _ in range(count):
            has_detection = random.random() < detection_rate
            metrics.append(self.generate_inference_metrics(model_name, has_detection))
        return metrics
