import logging


class Battery:
    """Simulates a battery with charging and discharging using Wh units."""

    def __init__(self, capacity_wh: float = 5.0, charge_rate_watts: float = 100.0):
        self.capacity_wh = capacity_wh
        self.charge_rate_watts = charge_rate_watts
        self.current_level_wh = capacity_wh  # Start fully charged
        self.logger = logging.getLogger(__name__)
        self.total_energy_used_wh = 0.0
        self.total_clean_energy_used_wh = 0.0

    def discharge(
        self,
        power_mw: float,
        duration_seconds: float,
        clean_energy_percentage: float = 0.0,
    ) -> bool:
        """
        Discharge battery by specified power for duration.

        Args:
            power_mw: Power consumption in milliwatts
            duration_seconds: Duration in seconds
            clean_energy_percentage: Percentage of energy from clean sources (0-100)

        Returns:
            True if discharge successful, False if insufficient battery
        """
        power_w = power_mw / 1000.0  # Convert mW to W
        energy_wh = power_w * (duration_seconds / 3600.0)  # Convert to Wh

        if self.current_level_wh < energy_wh:
            self.logger.error(
                f"Insufficient battery: {self.current_level_wh:.4f}Wh < {energy_wh:.4f}Wh"
            )
            return False

        self.current_level_wh -= energy_wh
        self.total_energy_used_wh += energy_wh
        self.total_clean_energy_used_wh += energy_wh * (clean_energy_percentage / 100.0)
        return True

    def charge(self, duration_seconds: float) -> float:
        """
        Charge battery for specified duration.

        Args:
            duration_seconds: Duration in seconds

        Returns:
            Actual energy added in Wh
        """
        energy_wh = self.charge_rate_watts * (duration_seconds / 3600.0)

        space_available = self.capacity_wh - self.current_level_wh
        actual_charge = min(energy_wh, space_available)

        self.current_level_wh += actual_charge
        return actual_charge

    def get_percentage(self) -> float:
        """Get current battery level as percentage."""
        return (self.current_level_wh / self.capacity_wh) * 100.0

    def get_level_wh(self) -> float:
        """Get current battery level in Wh."""
        return self.current_level_wh

    def get_total_energy_used_wh(self) -> float:
        """Get total energy used in Wh."""
        return self.total_energy_used_wh

    def get_total_clean_energy_used_wh(self) -> float:
        """Get total clean energy used in Wh."""
        return self.total_clean_energy_used_wh

    def reset_energy_tracking(self):
        """Reset energy usage tracking."""
        self.total_energy_used_wh = 0.0
        self.total_clean_energy_used_wh = 0.0
