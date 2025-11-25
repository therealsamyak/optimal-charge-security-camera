"""Configuration management utilities."""

import json
import re
from typing import Dict, Any
from loguru import logger


def load_config(config_path: str = "config.jsonc") -> Dict[str, Any]:
    """Load configuration from JSONC file."""
    try:
        with open(config_path, "r") as f:
            content = f.read()

            # Remove single-line comments (// ...)
            # But preserve // inside strings
            lines = []
            for line in content.split("\n"):
                # Simple approach: remove // comments that aren't in strings
                # For simplicity, just remove // and everything after on each line
                # This works if // isn't used in string values
                if "//" in line:
                    # Find // that's not inside quotes
                    parts = line.split("//", 1)
                    if len(parts) == 2:
                        # Count quotes before // to see if we're in a string
                        before_comment = parts[0]
                        quote_count = before_comment.count('"') - before_comment.count(
                            '\\"'
                        )
                        if quote_count % 2 == 0:  # Even number means not in string
                            line = parts[0].rstrip()
                lines.append(line)

            json_content = "\n".join(lines)

            # Remove trailing commas before } or ]
            json_content = re.sub(r",(\s*[}\]])", r"\1", json_content)

            config = json.loads(json_content)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        logger.error(
            f"Content preview: {content[:500] if 'content' in locals() else 'N/A'}"
        )
        raise
