"""
Config package for configuration management.

This package provides configuration loading and validation functionality.
"""

from .manager import ConfigManager, config

__all__ = ["ConfigManager", "config"]
