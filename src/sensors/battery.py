"""
Battery sensor interface for colleague to implement.

This module provides the interface for monitoring battery percentage and consumption.
Colleague should implement the actual hardware integration.
"""

from abc import ABC, abstractmethod


class BatterySensor(ABC):
    """Abstract interface for battery monitoring."""

    @abstractmethod
    def get_battery_percentage(self) -> float:
        """
        Get current battery percentage (0-100).

        Returns:
            float: Current battery level as percentage
        """
        pass

    @abstractmethod
    def consume_battery(self, amount: float) -> None:
        """
        Consume battery amount.

        Args:
            amount: Battery amount to consume (0-100)
        """
        pass

    @abstractmethod
    def is_charging(self) -> bool:
        """
        Check if battery is currently charging.

        Returns:
            bool: True if charging, False otherwise
        """
        pass

    @abstractmethod
    def start_charging(self) -> None:
        """Start charging the battery."""
        pass

    @abstractmethod
    def stop_charging(self) -> None:
        """Stop charging the battery."""
        pass

    @abstractmethod
    def get_battery_consumption_rate(self, model_name: str) -> float:
        """
        Get battery consumption rate for specific model.

        Args:
            model_name: Name of the model (e.g., 'yolov10n')

        Returns:
            float: Battery consumption rate per inference
        """
        pass
