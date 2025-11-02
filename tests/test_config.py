"""
Unit tests for configuration management.

Tests the ConfigManager class and YAML configuration loading.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config.manager import ConfigManager


class TestConfigManager:
    """Test cases for ConfigManager class."""

    def test_default_config_loading(self):
        """Test loading default configuration when no file exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "nonexistent.yml"
            manager = ConfigManager(str(config_path))

            # Should load defaults without error
            assert manager.get("source.type") == "image"
            assert manager.get("requirements.min_accuracy") == 80.0
            assert manager.get("models.default") == "yolov10n"

    def test_yaml_config_loading(self):
        """Test loading configuration from YAML file."""
        test_config = {
            "source": {"type": "webcam", "path": "/dev/video0"},
            "requirements": {"min_accuracy": 90.0, "max_latency_ms": 50.0},
            "controller": {"enable_charging": False},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.dump(test_config, f)
            config_path = f.name

        try:
            manager = ConfigManager(config_path)

            # Test loaded values
            assert manager.get("source.type") == "webcam"
            assert manager.get("source.path") == "/dev/video0"
            assert manager.get("requirements.min_accuracy") == 90.0
            assert manager.get("requirements.max_latency_ms") == 50.0
            assert manager.get("controller.enable_charging") is False

            # Test default values for missing keys
            assert manager.get("requirements.run_frequency_ms") == 2000.0
            assert manager.get("models.default") == "yolov10n"

        finally:
            os.unlink(config_path)

    def test_invalid_config_validation(self):
        """Test validation of invalid configuration values."""
        # Test invalid accuracy
        invalid_config = {
            "requirements": {
                "min_accuracy": 150.0  # Invalid: > 100
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.dump(invalid_config, f)
            config_path = f.name

        try:
            with pytest.raises(
                ValueError, match="min_accuracy must be between 0 and 100"
            ):
                ConfigManager(config_path)
        finally:
            os.unlink(config_path)

    def test_get_with_default(self):
        """Test getting configuration values with defaults."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "nonexistent.yml"
            manager = ConfigManager(str(config_path))

            # Test existing key
            assert manager.get("source.type") == "image"

            # Test non-existent key with default
            assert manager.get("nonexistent.key", "default_value") == "default_value"
            assert manager.get("nonexistent.key", 42) == 42
            assert manager.get("nonexistent.key") is None

    def test_get_all_config(self):
        """Test getting entire configuration dictionary."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "nonexistent.yml"
            manager = ConfigManager(str(config_path))

            all_config = manager.get_all()

            # Should contain all expected top-level keys
            expected_keys = [
                "source",
                "output",
                "runtime",
                "requirements",
                "controller",
                "models",
                "sensors",
            ]
            for key in expected_keys:
                assert key in all_config

            # Should be a copy (modifications shouldn't affect internal state)
            all_config["test"] = "value"
            assert manager.get("test") is None

    def test_config_reload(self):
        """Test reloading configuration from file."""
        initial_config = {"requirements": {"min_accuracy": 75.0}}

        updated_config = {"requirements": {"min_accuracy": 95.0}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.dump(initial_config, f)
            config_path = f.name

        try:
            manager = ConfigManager(config_path)
            assert manager.get("requirements.min_accuracy") == 75.0

            # Update file
            with open(config_path, "w") as f:
                yaml.dump(updated_config, f)

            # Reload and check updated value
            manager.reload()
            assert manager.get("requirements.min_accuracy") == 95.0

        finally:
            os.unlink(config_path)

    def test_save_config(self):
        """Test saving configuration to file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "nonexistent.yml"
            manager = ConfigManager(str(config_path))

            # Modify some values
            # Note: This would require adding setter methods to ConfigManager
            # For now, just test saving existing config
            save_path = Path(temp_dir) / "saved_config.yml"
            manager.save_config(str(save_path))

            # Verify file was created and contains valid YAML
            assert save_path.exists()

            with open(save_path, "r") as f:
                saved_config = yaml.safe_load(f)

            assert "source" in saved_config
            assert "requirements" in saved_config
