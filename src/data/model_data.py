"""YOLO model data loader for simulation."""

import pandas as pd
from typing import Dict, Any, Optional
from loguru import logger

# Module-level cache for model data (shared across all instances)
_model_data_cache: Optional[Dict[str, Dict[str, float]]] = None
_cache_config_hash: Optional[str] = None


class ModelDataLoader:
    """Loads and processes YOLO model performance data."""

    def __init__(
        self,
        csv_path: str = "datasets/model-data.csv",
        config: Optional[Dict[str, Any]] = None,
    ):
        self.csv_path = csv_path
        self.config = config or {}
        self.model_data = None
        self.load_data()

    def load_data(self) -> None:
        """Load YOLO model data from CSV."""
        try:
            import time

            start_time = time.time()
            self.model_data = pd.read_csv(self.csv_path)
            load_time = time.time() - start_time
            logger.info(
                f"Loaded model data for {len(self.model_data)} models "
                f"from {self.csv_path} in {load_time:.2f}s"
            )
        except Exception as e:
            logger.error(f"Failed to load model data from {self.csv_path}: {e}")
            raise

    def get_model_data(self) -> Dict[str, Dict[str, float]]:
        """Get complete model data with energy consumption rates.

        Uses module-level caching to avoid reprocessing the same data.
        Cache is keyed by config hash (energy rates may vary).
        """
        global _model_data_cache, _cache_config_hash

        import hashlib
        import json

        if self.model_data is None:
            return {}

        # Create config hash for cache key
        energy_rates = self.config.get("model_energy_consumption", {})
        config_hash = hashlib.md5(
            json.dumps(energy_rates, sort_keys=True).encode()
        ).hexdigest()

        # Check cache
        if _model_data_cache is not None and _cache_config_hash == config_hash:
            logger.debug("Using cached model data")
            return _model_data_cache

        # Cache miss - process data
        logger.info("Processing model data (cache miss)")
        import time

        start_time = time.time()

        result = {}
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

        process_time = time.time() - start_time
        logger.info(f"Processed {len(result)} models in {process_time:.2f}s")

        # Update cache
        _model_data_cache = result
        _cache_config_hash = config_hash

        return result
