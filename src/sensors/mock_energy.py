"""
Mock energy sensor implementation for testing.

This module provides a mock implementation of EnergySensor for testing purposes.
Colleague should replace this with actual hardware implementation.
"""

import random
import time
from .energy import EnergySensor


class MockEnergySensor(EnergySensor):
    """Mock implementation of energy sensor for testing."""

    def __init__(self):
        """Initialize mock energy sensor."""
        self._time_offset = time.time()
        self._energy_sources = ["solar", "wind", "hydro", "grid"]
        self._current_source = "grid"

    def get_energy_cleanliness(self) -> float:
        """
        Get current energy cleanliness percentage.

        Simulates varying energy cleanliness based on time of day and source.
        """
        current_time = time.time() - self._time_offset

        # Simulate daily patterns for renewable energy
        hour_of_day = (current_time / 3600) % 24

        if 6 <= hour_of_day <= 18:  # Daylight hours
            # Solar available, higher cleanliness
            base_cleanliness = 70.0 + random.uniform(-10, 20)
        else:  # Night hours
            # Less renewable energy, lower cleanliness
            base_cleanliness = 30.0 + random.uniform(-10, 15)

        # Add random fluctuations
        fluctuation = random.uniform(-5, 5)
        return max(0.0, min(100.0, base_cleanliness + fluctuation))

    def is_clean_energy_available(self) -> bool:
        """Check if clean energy is currently available."""
        return self.get_energy_cleanliness() > 60.0

    def get_energy_source_type(self) -> str:
        """Get current energy source type."""
        current_time = time.time() - self._time_offset
        hour_of_day = (current_time / 3600) % 24

        # Simulate energy source switching based on time and availability
        if 6 <= hour_of_day <= 18 and self.is_clean_energy_available():
            if random.random() < 0.7:
                self._current_source = "solar"
            elif random.random() < 0.5:
                self._current_source = "wind"
            else:
                self._current_source = "hydro"
        else:
            self._current_source = "grid"

        return self._current_source

    def get_energy_cost(self) -> float:
        """
        Get current energy cost per unit.

        Simulates varying energy costs based on source and time.
        """
        source = self.get_energy_source_type()

        # Base costs per source (lower is better)
        base_costs = {"solar": 0.05, "wind": 0.06, "hydro": 0.04, "grid": 0.12}

        base_cost = base_costs.get(source, 0.10)

        # Add time-based demand pricing
        current_time = time.time() - self._time_offset
        hour_of_day = (current_time / 3600) % 24

        if 18 <= hour_of_day <= 22:  # Peak hours
            multiplier = 1.5
        elif 6 <= hour_of_day <= 9:  # Morning peak
            multiplier = 1.2
        else:  # Off-peak
            multiplier = 0.8

        return base_cost * multiplier

    def simulate_energy_improvement(self) -> None:
        """Simulate sudden improvement in energy cleanliness (for testing)."""
        self._time_offset = time.time() - 8 * 3600  # Jump to good energy time
