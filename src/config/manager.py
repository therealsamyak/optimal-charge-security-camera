"""
Configuration management module for OCS Camera.

This module handles loading and validating configuration from YAML files.
"""

import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from loguru import logger


class ConfigManager:
    """Manages configuration loading and validation."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        if config_path is None:
            # Default to src/config/configuration.yml
            self.config_path = Path(__file__).parent / "configuration.yml"
        else:
            self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self.load_config()

    def load_config(self) -> None:
        """Load configuration from YAML file."""
        try:
            if not self.config_path.exists():
                logger.warning(
                    f"Config file not found: {self.config_path}. Using defaults."
                )
                self._config = self._get_default_config()
                return

            with open(self.config_path, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f) or {}

            # Validate and merge with defaults
            self._config = self._validate_config(self._config)
            logger.info(f"Configuration loaded from: {self.config_path}")

        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            logger.info("Using default configuration")
            self._config = self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "source": {"type": "webcam"},
            "output": {
                "runs_dir": "src/data/output/runs",
                "csv_file": "src/data/output/metrics.csv",
            },
            "runtime": {"interval_ms": 2000, "verbose": False},
            "requirements": {
                "min_accuracy": 80.0,
                "max_latency_ms": 100.0,
                "run_frequency_ms": 2000,
            },
            "controller": {
                "enable_charging": True,
                "min_battery_threshold": 20.0,
                "max_battery_threshold": 90.0,
            },
            "models": {
                "available": [
                    "yolov10n",
                    "yolov10s",
                    "yolov10m",
                    "yolov10b",
                    "yolov10l",
                    "yolov10x",
                ],
                "default": "yolov10n",
            },
            "sensors": {"use_mock": True, "mock_initial_battery": 80.0},
        }

    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and merge configuration with defaults."""
        default = self._get_default_config()
        merged = default.copy()

        # Deep merge
        for key, value in config.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                merged[key].update(value)
            else:
                merged[key] = value

        # Validate specific values
        if not 0 <= merged["requirements"]["min_accuracy"] <= 100:
            raise ValueError("min_accuracy must be between 0 and 100")

        if merged["requirements"]["max_latency_ms"] <= 0:
            raise ValueError("max_latency_ms must be positive")

        if merged["requirements"]["run_frequency_ms"] <= 0:
            raise ValueError("run_frequency_ms must be positive")

        if not 0 <= merged["controller"]["min_battery_threshold"] <= 100:
            raise ValueError("min_battery_threshold must be between 0 and 100")

        if not 0 <= merged["controller"]["max_battery_threshold"] <= 100:
            raise ValueError("max_battery_threshold must be between 0 and 100")

        return merged

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key (supports dot notation).

        Args:
            key: Configuration key (e.g., 'source.type')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self._config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def get_all(self) -> Dict[str, Any]:
        """Get entire configuration dictionary."""
        return self._config.copy()

    def reload(self) -> None:
        """Reload configuration from file."""
        self.load_config()

    def save_config(self, path: Optional[str] = None) -> None:
        """
        Save current configuration to file.

        Args:
            path: Path to save configuration. If None, uses current config path.
        """
        save_path = Path(path) if path else self.config_path

        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                yaml.dump(self._config, f, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to: {save_path}")
        except Exception as e:
            logger.error(f"Failed to save config to {save_path}: {e}")


# Global config instance
config = ConfigManager()
