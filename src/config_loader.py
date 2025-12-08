import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import re


class SimulationConfig:
    """Configuration for simulation parameters."""

    def __init__(
        self,
        duration_days: int = 7,
        task_interval_seconds: int = 5,
        time_acceleration: int = 1,
        user_accuracy_requirement: float = 45.0,
        user_latency_requirement: float = 8.0,
        battery_capacity_wh: float = 5.0,
        charge_rate_watts: float = 100.0,
        locations: Optional[List[str]] = None,
        seasons: Optional[List[str]] = None,
    ):
        self.duration_days = duration_days
        self.task_interval_seconds = task_interval_seconds
        self.time_acceleration = time_acceleration
        self.user_accuracy_requirement = user_accuracy_requirement
        self.user_latency_requirement = user_latency_requirement
        self.battery_capacity_wh = battery_capacity_wh
        self.charge_rate_watts = charge_rate_watts
        self.locations = locations or ["CA", "FL", "NW", "NY"]
        self.seasons = seasons or ["winter", "spring", "summer", "fall"]


class BatchConfig:
    """Configuration for batch simulation parameters."""

    def __init__(
        self,
        num_variations: int = 10,
        random_seed: Optional[int] = None,
        output_detailed_csv: bool = True,
        accuracy_range: Optional[Dict[str, float]] = None,
        latency_range: Optional[Dict[str, float]] = None,
        battery_capacity_range: Optional[Dict[str, float]] = None,
        charge_rate_range: Optional[Dict[str, float]] = None,
    ):
        self.num_variations = num_variations
        self.random_seed = random_seed
        self.output_detailed_csv = output_detailed_csv
        self.accuracy_range = accuracy_range or {"min": 30, "max": 80}
        self.latency_range = latency_range or {"min": 2, "max": 15}
        self.battery_capacity_range = battery_capacity_range or {"min": 2, "max": 15}
        self.charge_rate_range = charge_rate_range or {"min": 50, "max": 200}


class WorkersConfig:
    """Configuration for parallel processing workers."""

    def __init__(self, max_workers: int = 100):
        self.max_workers = max_workers


class ConfigLoader:
    """Loads and validates simulation configuration."""

    def __init__(self, config_path: str = "config.jsonc"):
        self.config_path = Path(config_path)
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file, ignoring comments."""
        try:
            with open(self.config_path, "r") as f:
                content = f.read()
                # Remove JSONC comments (// and /* */)
                content = re.sub(r"//.*", "", content)
                content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
                config = json.loads(content)
            self.logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except FileNotFoundError:
            self.logger.error(f"Configuration file {self.config_path} not found")
            return self._get_default_config()
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in configuration file: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "simulation": {
                "duration_days": 7,
                "task_interval_seconds": 5,
                "time_acceleration": 1,
                "user_accuracy_requirement": 45.0,
                "user_latency_requirement": 8.0,
            },
            "battery": {"capacity_wh": 5.0, "charge_rate_watts": 100},
            "locations": ["CA", "FL", "NW", "NY"],
            "seasons": ["winter", "spring", "summer", "fall"],
            "controllers": ["naive_weak", "naive_strong", "oracle", "custom"],
            "output_dir": "results/",
            "batch_run": {
                "num_variations": 10,
                "random_seed": None,
                "output_detailed_csv": True,
                "accuracy_range": {"min": 30, "max": 80},
                "latency_range": {"min": 2, "max": 15},
                "battery_capacity_range": {"min": 2, "max": 15},
                "charge_rate_range": {"min": 50, "max": 200},
            },
        }

    def get_simulation_config(self) -> SimulationConfig:
        """Get simulation configuration as SimulationConfig object."""
        sim_config = self.config.get("simulation", {})
        battery_config = self.config.get("battery", {})

        return SimulationConfig(
            duration_days=sim_config.get("duration_days", 7),
            task_interval_seconds=sim_config.get("task_interval_seconds", 5),
            time_acceleration=sim_config.get("time_acceleration", 1),
            user_accuracy_requirement=sim_config.get("user_accuracy_requirement", 45.0),
            user_latency_requirement=sim_config.get("user_latency_requirement", 8.0),
            battery_capacity_wh=battery_config.get("capacity_wh", 5.0),
            charge_rate_watts=battery_config.get("charge_rate_watts", 100),
            locations=self.config.get("locations", ["CA", "FL", "NW", "NY"]),
            seasons=self.config.get("seasons", ["winter", "spring", "summer", "fall"]),
        )

    def get_locations(self) -> List[str]:
        """Get list of locations."""
        return self.config.get("locations", ["CA", "FL", "NW", "NY"])

    def get_seasons(self) -> List[str]:
        """Get list of seasons."""
        return self.config.get("seasons", ["winter", "spring", "summer", "fall"])

    def get_controllers(self) -> List[str]:
        """Get list of controller types."""
        return self.config.get(
            "controllers", ["naive_weak", "naive_strong", "oracle", "custom"]
        )

    def get_output_dir(self) -> str:
        """Get output directory."""
        return self.config.get("output_dir", "results/")

    def get_batch_config(self) -> BatchConfig:
        """Get batch configuration as BatchConfig object."""
        batch_config = self.config.get("batch_run", {})

        return BatchConfig(
            num_variations=batch_config.get("num_variations", 10),
            random_seed=batch_config.get("random_seed"),
            output_detailed_csv=batch_config.get("output_detailed_csv", True),
            accuracy_range=batch_config.get("accuracy_range", {"min": 30, "max": 80}),
            latency_range=batch_config.get("latency_range", {"min": 2, "max": 15}),
            battery_capacity_range=batch_config.get(
                "battery_capacity_range", {"min": 2, "max": 15}
            ),
            charge_rate_range=batch_config.get(
                "charge_rate_range", {"min": 50, "max": 200}
            ),
        )

    def get_workers_config(self) -> WorkersConfig:
        """Get workers configuration as WorkersConfig object."""
        workers_config = self.config.get("workers", {})

        return WorkersConfig(max_workers=workers_config.get("max_workers", 100))

    def validate_config(self) -> bool:
        """Validate configuration parameters."""
        try:
            sim_config = self.config.get("simulation", {})
            battery_config = self.config.get("battery", {})

            # Validate simulation config
            if sim_config.get("duration_days", 0) <= 0:
                self.logger.error("duration_days must be positive")
                return False

            if sim_config.get("task_interval_seconds", 0) <= 0:
                self.logger.error("task_interval_seconds must be positive")
                return False

            if sim_config.get("time_acceleration", 0) <= 0:
                self.logger.error("time_acceleration must be positive")
                return False

            # Validate battery config
            if battery_config.get("capacity_wh", 0) <= 0:
                self.logger.error("capacity_wh must be positive")
                return False

            if battery_config.get("charge_rate_watts", 0) <= 0:
                self.logger.error("charge_rate_watts must be positive")
                return False

            # Validate lists
            required_lists = ["locations", "seasons", "controllers"]
            for list_name in required_lists:
                if not isinstance(self.config.get(list_name, []), list):
                    self.logger.error(f"{list_name} must be a list")
                    return False
                if len(self.config.get(list_name, [])) == 0:
                    self.logger.error(f"{list_name} cannot be empty")
                    return False

            # Validate batch config if present
            batch_config = self.config.get("batch_run", {})
            if batch_config:
                if batch_config.get("num_variations", 0) <= 0:
                    self.logger.error("num_variations must be positive")
                    return False

                # Validate ranges
                range_keys = [
                    "accuracy_range",
                    "latency_range",
                    "battery_capacity_range",
                    "charge_rate_range",
                ]
                for range_key in range_keys:
                    range_val = batch_config.get(range_key, {})
                    if not isinstance(range_val, dict):
                        self.logger.error(f"{range_key} must be a dict")
                        return False
                    if range_val.get("min", 0) < 0 or range_val.get("max", 0) <= 0:
                        self.logger.error(f"{range_key} values must be positive")
                        return False
                    if range_val.get("min", 0) >= range_val.get("max", 0):
                        self.logger.error(f"{range_key} min must be less than max")
                        return False

            self.logger.info("Configuration validation passed")
            return True

        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False
