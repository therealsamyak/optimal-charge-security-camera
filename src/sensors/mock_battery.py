"""
Mock battery sensor implementation for testing.

This module provides a mock implementation of BatterySensor for testing purposes.
Colleague should replace this with actual hardware implementation.
"""

import random
import time
from typing import Dict, List
from .battery import BatterySensor


class MockBatterySensor(BatterySensor):
    """Mock implementation of battery sensor for testing."""

    def __init__(self, initial_battery: float = 80.0):
        """
        Initialize mock battery sensor.

        Args:
            initial_battery: Starting battery percentage (0-100)
        """
        self._battery_level = max(0.0, min(100.0, initial_battery))
        self._is_charging = False
        self._consumption_rates = {
            "yolov10n": 0.1,  # Smallest model, lowest consumption
            "yolov10s": 0.2,
            "yolov10m": 0.4,
            "yolov10b": 0.6,
            "yolov10l": 0.8,
            "yolov10x": 1.0,  # Largest model, highest consumption
        }

    def get_battery_percentage(self) -> float:
        """Get current battery percentage with small random fluctuation."""
        # Add small random fluctuation to simulate real sensor
        fluctuation = random.uniform(-0.5, 0.5)
        return max(0.0, min(100.0, self._battery_level + fluctuation))

    def consume_battery(self, amount: float) -> None:
        """Consume specified battery amount."""
        self._battery_level = max(0.0, self._battery_level - amount)

    def is_charging(self) -> bool:
        """Check if currently charging."""
        return self._is_charging

    def start_charging(self) -> None:
        """Start charging the battery."""
        self._is_charging = True

    def stop_charging(self) -> None:
        """Stop charging the battery."""
        self._is_charging = False

    def get_battery_consumption_rate(self, model_name: str) -> float:
        """Get battery consumption rate for specific model."""
        return self._consumption_rates.get(model_name, 0.5)

    def get_consumption_history(self) -> List[Dict]:
        """Get history of battery consumption events."""
        return getattr(self, "_consumption_history", [])

    def track_consumption(self, model_name: str, amount: float) -> None:
        """
        Track battery consumption event.

        Args:
            model_name: Model that consumed battery
            amount: Amount consumed
        """
        if not hasattr(self, "_consumption_history"):
            self._consumption_history = []

        self._consumption_history.append(
            {
                "timestamp": time.time(),
                "model_name": model_name,
                "consumption_amount": amount,
                "battery_level_before": self._battery_level + amount,
                "battery_level_after": self._battery_level,
            }
        )

        # Keep only last 1000 events
        if len(self._consumption_history) > 1000:
            self._consumption_history = self._consumption_history[-1000:]

    def get_consumption_stats(self) -> Dict:
        """Get battery consumption statistics."""
        history = self.get_consumption_history()
        if not history:
            return {}

        # Group by model
        model_stats = {}
        total_consumption = 0.0

        for event in history:
            model = event["model_name"]
            amount = event["consumption_amount"]

            if model not in model_stats:
                model_stats[model] = {
                    "count": 0,
                    "total_consumption": 0.0,
                    "avg_consumption": 0.0,
                }

            model_stats[model]["count"] += 1
            model_stats[model]["total_consumption"] += amount
            total_consumption += amount

        # Calculate averages
        for model in model_stats:
            model_stats[model]["avg_consumption"] = (
                model_stats[model]["total_consumption"] / model_stats[model]["count"]
            )

        return {
            "total_events": len(history),
            "total_consumption": total_consumption,
            "model_breakdown": model_stats,
            "consumption_rate_per_hour": total_consumption
            * (3600 / (history[-1]["timestamp"] - history[0]["timestamp"]))
            if len(history) > 1
            else 0.0,
        }

    def simulate_charging(self, charging_rate: float = 2.0) -> None:
        """
        Simulate battery charging (for testing).

        Args:
            charging_rate: Battery percentage increase per second
        """
        if self._is_charging:
            self._battery_level = min(100.0, self._battery_level + charging_rate)

    def simulate_time_passage(self, seconds: float) -> None:
        """
        Simulate natural battery drain over time.

        Args:
            seconds: Time passed in seconds
        """
        # Natural drain rate: 0.1% per hour when idle
        natural_drain = (0.1 / 3600.0) * seconds
        self._battery_level = max(0.0, self._battery_level - natural_drain)
