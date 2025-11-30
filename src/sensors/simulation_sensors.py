"""Enhanced mock sensors for simulation."""

from typing import Dict, Any
from loguru import logger

from src.utils.energy import energy_percent_to_wh, energy_wh_to_percent


class SimulationSensors:
    """Enhanced mock sensors for battery and energy simulation."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.battery_config = config.get("battery", {})

        # Get battery capacity in Wh (canonical source: battery.capacity_wh)
        # Default: 4.0 Wh (aligned with config.jsonc default)
        self._capacity_wh = float(self.battery_config.get("capacity_wh", 4.0))

        # Get initial capacity as percentage (0-100)
        # Canonical: battery.initial_capacity_pct, fallback: battery.initial_capacity (legacy)
        initial_capacity_pct = self.battery_config.get(
            "initial_capacity_pct", 
            self.battery_config.get("initial_capacity", 100.0)
        )
        
        # Get low battery threshold as percentage (0-100)
        # Canonical: battery.low_battery_threshold_pct, fallback: battery.low_battery_threshold (legacy)
        low_battery_threshold_pct = self.battery_config.get(
            "low_battery_threshold_pct",
            self.battery_config.get("low_battery_threshold", 20.0)
        )

        # Get charging rate in Watts (canonical source: battery.charging_rate_watts)
        # Default: 0.5 W (for 4 Wh battery, gives ~8 hour full charge)
        charging_rate_watts = float(self.battery_config.get("charging_rate_watts", 0.5))
        
        # Derive charging_rate (percent per second) from physical charging power
        # Formula: charging_rate (%/s) = (charging_rate_watts / capacity_wh) * 100 / 3600
        # This converts Watts -> Wh/s -> %/s
        if self._capacity_wh > 0:
            charging_rate_pct_per_sec = (charging_rate_watts / self._capacity_wh) * 100.0 / 3600.0
        else:
            logger.warning(f"Invalid battery capacity {self._capacity_wh} Wh, using fallback charging rate")
            charging_rate_pct_per_sec = 0.0035  # Legacy default
        
        # Fallback to legacy charging_rate if new config not available
        legacy_charging_rate = self.battery_config.get("charging_rate")
        if legacy_charging_rate is not None:
            charging_rate_pct_per_sec = float(legacy_charging_rate)

        # Battery state (stored as percentage 0-100 for now, to maintain external API)
        self.current_capacity = float(initial_capacity_pct)
        self.max_capacity = 100.0  # Always 100% (max is defined by capacity_wh)
        self.charging_rate = charging_rate_pct_per_sec
        self.low_battery_threshold = float(low_battery_threshold_pct)

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

    def _get_capacity_wh(self) -> float:
        """Get battery capacity in Watt-hours (internal helper for future refactoring)."""
        return self._capacity_wh

    def _percent_to_wh(self, percent: float) -> float:
        """Convert battery percentage (0-100) to Watt-hours (internal helper).
        
        Args:
            percent: Battery percentage (0-100)
            
        Returns:
            Energy in Watt-hours
        """
        return energy_percent_to_wh(percent, self._capacity_wh)

    def _wh_to_percent(self, energy_wh: float) -> float:
        """Convert energy in Watt-hours to battery percentage (0-100) (internal helper).
        
        Args:
            energy_wh: Energy in Watt-hours
            
        Returns:
            Battery percentage (0-100)
        """
        return energy_wh_to_percent(energy_wh, self._capacity_wh)
