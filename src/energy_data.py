import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


class EnergyData:
    """Handles loading and processing of clean energy data."""

    def __init__(self, data_dir: str = "energy-data"):
        self.data_dir = Path(data_dir)
        self.logger = logging.getLogger(__name__)
        self.data: Dict[str, pd.DataFrame] = {}
        self.load_all_data()

    def load_all_data(self):
        """Load all CSV files from energy data directory."""
        csv_files = list(self.data_dir.glob("*.csv"))

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, dtype=str, low_memory=False)
                region_name = csv_file.stem
                self.data[region_name] = df
                self.logger.info(f"Loaded {len(df)} records for {region_name}")
            except Exception as e:
                self.logger.error(f"Failed to load {csv_file}: {e}")

    def get_clean_energy_percentage(
        self, region: str, datetime_str: str
    ) -> Optional[float]:
        """
        Get clean energy percentage for a specific region and datetime.

        Args:
            region: Region identifier (e.g., "US-CAL-LDWP_2024_5_minute")
            datetime_str: Datetime string in format matching CSV

        Returns:
            Clean energy percentage (0-100) or None if not found
        """
        if region not in self.data:
            self.logger.error(f"Region {region} not found in energy data")
            return None

        df = self.data[region]

        # Convert datetime column to datetime objects if not already
        if "Datetime (UTC)" in df.columns:
            df["Datetime (UTC)"] = pd.to_datetime(df["Datetime (UTC)"])

            # Find the closest datetime not in the future
            target_dt = pd.to_datetime(datetime_str)
            past_data = df[df["Datetime (UTC)"] <= target_dt]

            if past_data.empty:
                self.logger.error(f"No past data available for {datetime_str}")
                return None

            # Get the most recent data point
            latest_row = past_data.iloc[-1]

            if "Carbon-free energy percentage (CFE%)" in latest_row:
                return float(latest_row["Carbon-free energy percentage (CFE%)"])

        self.logger.error(f"Clean energy data not found for {region} at {datetime_str}")
        return None

    def get_available_regions(self) -> List[str]:
        """Get list of available regions."""
        return list(self.data.keys())

    def get_data_for_day(self, region: str, date_str: str) -> pd.DataFrame:
        """
        Get all data for a specific day and region.

        Args:
            region: Region identifier
            date_str: Date string (YYYY-MM-DD format)

        Returns:
            DataFrame with data for the specified day
        """
        if region not in self.data:
            self.logger.error(f"Region {region} not found")
            return pd.DataFrame()

        df = self.data[region].copy()
        df["Datetime (UTC)"] = pd.to_datetime(df["Datetime (UTC)"])

        # Filter for specific date
        target_date = pd.to_datetime(date_str).date()
        day_data = df[df["Datetime (UTC)"].dt.date == target_date]

        return pd.DataFrame(day_data)
