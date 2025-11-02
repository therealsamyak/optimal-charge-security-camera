"""
Energy cleanliness sensor interface for colleague to implement.

This module provides the interface for monitoring energy source cleanliness.
Colleague should implement the actual energy monitoring integration.
"""

from abc import ABC, abstractmethod


class EnergySensor(ABC):
    """Abstract interface for energy cleanliness monitoring."""

    @abstractmethod
    def get_energy_cleanliness(self) -> float:
        """
        Get current energy cleanliness percentage (0-100).

        Returns:
            float: Energy cleanliness as percentage
        """
        pass

    @abstractmethod
    def is_clean_energy_available(self) -> bool:
        """
        Check if clean energy is currently available.

        Returns:
            bool: True if clean energy available, False otherwise
        """
        pass

    @abstractmethod
    def get_energy_source_type(self) -> str:
        """
        Get current energy source type.

        Returns:
            str: Type of energy source (e.g., 'solar', 'wind', 'grid')
        """
        pass

    @abstractmethod
    def get_energy_cost(self) -> float:
        """
        Get current energy cost per unit.

        Returns:
            float: Energy cost
        """
        pass
