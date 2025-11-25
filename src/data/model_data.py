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
            # Combine model and version to create model name (e.g., "YOLOv10" + "N" -> "YOLOv10-N")
            model_base = str(row["model"]).strip('"')
            version = str(row["version"]).strip('"')
            model_name = f"{model_base}-{version}"

            # Get energy consumption rate from config
            energy_rate = energy_rates.get(model_name, 0.01)  # Default fallback

            # Convert COCO mAP 50-95 from percentage to 0-1 scale (e.g., 39.5 -> 0.395)
            coco_map = float(str(row["COCO mAP 50-95"]).strip('"'))
            accuracy = coco_map / 100.0

            # Get latency in milliseconds
            latency_ms = float(
                str(row["Latency T4 TensorRT10 FP16(ms/img)"]).strip('"')
            )

            result[model_name] = {
                "accuracy": accuracy,
                "latency_ms": latency_ms,
                "energy_consumption": energy_rate,
            }

        return result
