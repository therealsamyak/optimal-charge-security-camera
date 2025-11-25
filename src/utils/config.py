"""Configuration management utilities."""

import json
from typing import Dict, Any
from loguru import logger


def load_config(config_path: str = "config.jsonc") -> Dict[str, Any]:
    """Load configuration from JSONC file."""
    try:
        with open(config_path, "r") as f:
            content = f.read()
            # Remove comments for JSON parsing
            lines = [
                line
                for line in content.split("\n")
                if not line.strip().startswith("//")
            ]
            json_content = "\n".join(lines)
            config = json.loads(json_content)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise
