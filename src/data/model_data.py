"""YOLO model data loader for simulation."""

import pandas as pd
from typing import Dict, Any, Optional
from loguru import logger

from src.utils.energy import (
    calculate_inference_energy_wh,
    energy_wh_to_percent,
)

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
        Cache is keyed by config hash (energy rates and physical params may vary).
        """
        global _model_data_cache, _cache_config_hash

        import hashlib
        import json
        import time

        if self.model_data is None:
            return {}

        # Config values used in energy calculation
        energy_rates = self.config.get("model_energy_consumption", {})
        battery_cfg = self.config.get("battery", {})
        
        # Get battery capacity from config (canonical source: battery.capacity_wh)
        # Default: 4.0 Wh (aligned with config.jsonc default)
        capacity_wh = float(battery_cfg.get("capacity_wh", 4.0))
        
        # Get device power from config (canonical source: device.power_watts)
        # Fallback chain: device.power_watts -> device_power_watts (legacy) -> 5.0 (default)
        device_cfg = self.config.get("device", {})
        if "power_watts" in device_cfg:
            device_power_watts = float(device_cfg["power_watts"])
        else:
            device_power_watts = float(self.config.get("device_power_watts", 5.0))

        # Create config hash for cache key
        config_hash_input = {
            "energy_rates": energy_rates,
            "capacity_wh": capacity_wh,
            "device_power_watts": device_power_watts,
        }
        config_hash = hashlib.md5(
            json.dumps(config_hash_input, sort_keys=True).encode()
        ).hexdigest()

        # Check cache
        if _model_data_cache is not None and _cache_config_hash == config_hash:
            logger.debug("Using cached model data")
            return _model_data_cache

        # Cache miss, process data
        logger.info("Processing model data (cache miss)")
        start_time = time.time()

        result: Dict[str, Dict[str, float]] = {}
        for _, row in self.model_data.iterrows():
            # Combine model and version to create model name (for example "YOLOv10" and "N" to "YOLOv10-N")
            model_base = str(row["model"]).strip('"')
            version = str(row["version"]).strip('"')
            model_name = f"{model_base}-{version}"

            # Convert COCO mAP 50-95 from percentage to 0 to 1 scale (for example 39.5 to 0.395)
            coco_map = float(str(row["COCO mAP 50-95"]).strip('"'))
            accuracy = coco_map / 100.0

            # Get latency in milliseconds
            latency_ms = float(
                str(row["Latency T4 TensorRT10 FP16(ms/img)"]).strip('"')
            )
            latency_s = latency_ms / 1000.0

            # Calculate energy per inference in Watt-hours using physical units
            energy_wh = calculate_inference_energy_wh(device_power_watts, latency_s)
            
            # Convert energy in Wh to battery percentage (0-100)
            if capacity_wh > 0:
                percent_per_inf = energy_wh_to_percent(energy_wh, capacity_wh)
            else:
                # Safe fallback if capacity is invalid (should not happen with proper config)
                logger.warning(f"Invalid battery capacity {capacity_wh} Wh, using fallback")
                percent_per_inf = 0.01  # Default: 0.01% per inference

            # Optional override from config if provided
            energy_override = energy_rates.get(model_name)
            energy_rate = energy_override if energy_override is not None else percent_per_inf

            if energy_override is not None:
                logger.debug(
                    f"Using override energy rate for {model_name}: {energy_rate:.6f} percent per inference"
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
