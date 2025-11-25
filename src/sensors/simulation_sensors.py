"""Enhanced mock sensors for simulation."""

from typing import Dict, Any


class SimulationSensors:
    """Enhanced mock sensors for battery and energy simulation."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.battery_config = config["battery"]

        # Battery state
        self.current_capacity = self.battery_config["initial_capacity"]
        self.max_capacity = self.battery_config["max_capacity"]
        self.charging_rate = self.battery_config["charging_rate"]
        self.low_battery_threshold = self.battery_config["low_battery_threshold"]

        # Energy state
        self.current_cleanliness = 0.5  # Default fallback

    def get_battery_level(self) -> float:
        """Get current battery level as percentage."""
        return self.current_capacity

    def is_charging(self) -> bool:
        """Check if battery is currently charging."""
        return self.current_capacity < self.low_battery_threshold

    def consume_energy(self, amount: float) -> bool:
        """Consume energy from battery."""
        if self.current_capacity >= amount:
            self.current_capacity -= amount
            return True
        return False

    def charge_battery(self, time_seconds: float) -> None:
        """Charge battery for given time period."""
        charge_amount = self.charging_rate * time_seconds
        self.current_capacity = min(
            self.max_capacity, self.current_capacity + charge_amount
        )

    def update_energy_cleanliness(self, cleanliness: float) -> None:
        """Update current energy cleanliness percentage."""
        self.current_cleanliness = max(0.0, min(1.0, cleanliness))

    def get_energy_cleanliness(self) -> float:
        """Get current energy cleanliness percentage."""
        return self.current_cleanliness

    def reset(self) -> None:
        """Reset sensors to initial state."""
        self.current_capacity = self.battery_config["initial_capacity"]
        self.current_cleanliness = 0.5
