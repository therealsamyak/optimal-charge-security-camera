"""
Model data loader for YOLO performance profiles.

This module loads model performance data from CSV and provides
access to model profiles with power consumption estimates.
"""

import csv
from typing import Dict, Optional
from dataclasses import dataclass
from loguru import logger
from pathlib import Path


@dataclass
class ModelProfile:
    """Performance profile for a YOLO model."""

    name: str
    version: str
    latency_ms: float
    accuracy: float  # COCO mAP 50-95
    battery_consumption: float  # Placeholder power consumption
    size_rank: int  # Size ranking for controller scoring


class ModelDataLoader:
    """Loader for YOLO model performance data from CSV."""

    def __init__(self, csv_path: Optional[str] = None):
        """
        Initialize model data loader.

        Args:
            csv_path: Path to model data CSV file
        """
        if csv_path is None:
            csv_path = str(Path(__file__).parent / "model-data.csv")

        self.csv_path = Path(csv_path)
        self.model_profiles: Dict[str, ModelProfile] = {}
        self._load_model_data()

    def _load_model_data(self) -> None:
        """Load model data from CSV file."""
        if not self.csv_path.exists():
            logger.error(f"Model data CSV not found: {self.csv_path}")
            return

        try:
            with open(self.csv_path, "r", encoding="utf-8") as file:
                reader = csv.DictReader(file)

                for row in reader:
                    model_name = row["model"].strip('"')
                    version = row["version"].strip('"')
                    latency = float(row["Latency T4 TensorRT10 FP16(ms/img)"])
                    accuracy = float(row["COCO mAP 50-95"])

                    # Create model key
                    model_key = f"{model_name.lower()}{version.lower()}"

                    # Determine size rank and power consumption
                    size_rank = self._get_size_rank(version)
                    battery_consumption = self._get_battery_consumption(version)

                    # Normalize accuracy to 60-100 range
                    normalized_accuracy = self._normalize_accuracy(accuracy)

                    profile = ModelProfile(
                        name=model_key,
                        version=version,
                        latency_ms=latency,
                        accuracy=normalized_accuracy,
                        battery_consumption=battery_consumption,
                        size_rank=size_rank,
                    )

                    self.model_profiles[model_key] = profile

            logger.info(
                f"Loaded {len(self.model_profiles)} model profiles from {self.csv_path}"
            )

        except Exception as e:
            logger.error(f"Failed to load model data: {e}")

    def _get_size_rank(self, version: str) -> int:
        """Get size rank based on model version."""
        size_mapping = {
            "t": 1,
            "n": 2,
            "s": 3,
            "m": 4,
            "l": 5,
            "b": 6,
            "c": 7,
            "e": 8,
            "x": 9,
        }
        return size_mapping.get(version.lower(), 5)

    def _get_battery_consumption(self, version: str) -> float:
        """Get placeholder battery consumption based on model size."""
        # Placeholder values - smaller models use less power
        consumption_mapping = {
            "t": 0.05,
            "n": 0.1,
            "s": 0.2,
            "m": 0.4,
            "l": 0.6,
            "b": 0.7,
            "c": 0.8,
            "e": 0.9,
            "x": 1.0,
        }
        return consumption_mapping.get(version.lower(), 0.5)

    def _normalize_accuracy(self, raw_accuracy: float) -> float:
        """Normalize accuracy from COCO mAP (0-100) to 60-95 range."""
        # YOLOv10 models range from ~39.5% to 54.4% COCO mAP
        # Map this range to 60-95 for user-facing accuracy
        min_raw = 39.5
        max_raw = 54.4
        min_normalized = 60.0
        max_normalized = 95.0

        # Linear normalization
        if raw_accuracy <= min_raw:
            return min_normalized
        elif raw_accuracy >= max_raw:
            return max_normalized
        else:
            # Scale to 60-100 range
            ratio = (raw_accuracy - min_raw) / (max_raw - min_raw)
            return min_normalized + ratio * (max_normalized - min_normalized)

    def get_profile(self, model_name: str) -> Optional[ModelProfile]:
        """Get profile for a specific model."""
        return self.model_profiles.get(model_name.lower())

    def get_all_profiles(self) -> Dict[str, ModelProfile]:
        """Get all model profiles."""
        return self.model_profiles.copy()

    def get_yolo_models_by_version(self, yolo_version: str) -> Dict[str, ModelProfile]:
        """Get all models for a specific YOLO version."""
        return {
            name: profile
            for name, profile in self.model_profiles.items()
            if name.startswith(yolo_version.lower())
        }

    def get_available_models(self) -> list:
        """Get list of available model names."""
        return list(self.model_profiles.keys())

    def filter_models_by_requirements(
        self, min_accuracy: float = 0.0, max_latency_ms: float = float("inf")
    ) -> Dict[str, ModelProfile]:
        """Filter models by performance requirements."""
        filtered = {}
        for name, profile in self.model_profiles.items():
            if (
                profile.accuracy >= min_accuracy
                and profile.latency_ms <= max_latency_ms
            ):
                filtered[name] = profile
        return filtered


# Global instance for easy access
_model_loader = None


def get_model_loader() -> ModelDataLoader:
    """Get global model loader instance."""
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelDataLoader()
    return _model_loader
