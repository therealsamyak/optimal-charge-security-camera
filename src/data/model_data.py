"""YOLO model data loader for simulation."""

import pandas as pd
from typing import Dict, Any, Optional
from loguru import logger


class ModelDataLoader:
    """Loads and processes YOLO model performance data."""

    def __init__(
        self, csv_path: str = "model-data.csv", config: Optional[Dict[str, Any]] = None
    ):
        self.csv_path = csv_path
        self.config = config or {}
        self.model_data = None
        self.load_data()

    def load_data(self) -> None:
        """Load YOLO model data from CSV."""
        try:
            self.model_data = pd.read_csv(self.csv_path)
            logger.info(f"Loaded model data for {len(self.model_data)} models")
        except Exception as e:
            logger.error(f"Failed to load model data: {e}")
            raise

    def get_model_data(self) -> Dict[str, Dict[str, float]]:
        """Get complete model data with energy consumption rates."""
        if self.model_data is None:
            return {}

        result = {}
        energy_rates = self.config.get("model_energy_consumption", {})

        for _, row in self.model_data.iterrows():
            model_name = row["model"]

            # Get energy consumption rate from config
            energy_rate = energy_rates.get(model_name, 0.01)  # Default fallback

            result[model_name] = {
                "accuracy": float(row["accuracy"]),
                "latency_ms": float(row["latency_ms"]),
                "energy_consumption": energy_rate,
            }

        return result

    def get_model_by_name(self, model_name: str) -> Dict[str, float]:
        """Get specific model data."""
        all_models = self.get_model_data()
        return all_models.get(model_name, {})

    def get_models_meeting_thresholds(
        self, accuracy_threshold: float, latency_threshold: float
    ) -> Dict[str, Dict[str, float]]:
        """Get models that meet performance thresholds."""
        all_models = self.get_model_data()
        valid_models = {}

        for model_name, data in all_models.items():
            if (
                data["accuracy"] >= accuracy_threshold
                and data["latency_ms"] <= latency_threshold
            ):
                valid_models[model_name] = data

        return valid_models
