"""LDWP energy data loader for simulation."""

import pandas as pd
from typing import Dict
from datetime import datetime, timedelta
from loguru import logger


class EnergyLoader:
    """Loads and processes LDWP carbon intensity data."""

    def __init__(self, csv_path: str = "US-CAL-LDWP_2024_5_minute.csv"):
        self.csv_path = csv_path
        self.data = None
        self.load_data()

    def load_data(self) -> None:
        """Load LDWP data from CSV."""
        try:
            self.data = pd.read_csv(self.csv_path)
            logger.info(f"Loaded {len(self.data)} records from {self.csv_path}")
        except Exception as e:
            logger.error(f"Failed to load energy data: {e}")
            raise

    def get_clean_energy_percentage(self, timestamp: datetime) -> float:
        """Get clean energy percentage for given timestamp."""
        if self.data is None:
            return 0.5  # Default fallback

        # Find closest 5-minute interval
        timestamp_rounded = timestamp.replace(second=0, microsecond=0)
        minute_mod = timestamp_rounded.minute % 5
        if minute_mod >= 3:
            timestamp_rounded += timedelta(minutes=5 - minute_mod)
        else:
            timestamp_rounded -= timedelta(minutes=minute_mod)

        # Filter data for matching timestamp
        matching_data = self.data[
            (pd.to_datetime(self.data["Datetime (UTC)"]) == timestamp_rounded)
        ]

        if matching_data.empty:
            return 0.5  # Default fallback

        # Convert carbon intensity to clean energy percentage
        # Lower carbon intensity = higher clean energy percentage
        carbon_intensity = matching_data["Carbon intensity gCO₂eq/kWh (direct)"].iloc[0]

        # Simple conversion: 0 gCO2/kWh = 100% clean, 1000 gCO2/kWh = 0% clean
        clean_percentage = max(0.0, min(1.0, 1.0 - (carbon_intensity / 1000.0)))

        return clean_percentage

    def get_seasonal_day_data(self, date: str) -> Dict[datetime, float]:
        """Get full day energy data for seasonal simulation."""
        target_date = datetime.strptime(date, "%Y-%m-%d").date()

        # Filter data for the target date
        daily_data = {}
        for _, row in self.data.iterrows():
            timestamp = pd.to_datetime(row["Datetime (UTC)"])
            if timestamp.date() == target_date:
                carbon_intensity = row["Carbon intensity gCO₂eq/kWh (direct)"]
                clean_percentage = max(0.0, min(1.0, 1.0 - (carbon_intensity / 1000.0)))
                daily_data[timestamp] = clean_percentage

        return daily_data
