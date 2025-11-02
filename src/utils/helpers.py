"""
Shared utilities for OCS Camera.

This module provides common utility functions used across the application.
"""

import csv
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from loguru import logger


def ensure_directory(path: str) -> None:
    """
    Ensure directory exists, create if it doesn't.

    Args:
        path: Directory path to ensure exists
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def write_csv_log(
    csv_path: str, data: Dict[str, Any], headers: Optional[List[str]] = None
) -> None:
    """
    Write data to CSV log file.

    Args:
        csv_path: Path to CSV file
        data: Dictionary of data to write
        headers: List of headers (if file doesn't exist)
    """
    ensure_directory(os.path.dirname(csv_path))

    file_exists = os.path.exists(csv_path)

    try:
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())

            if not file_exists or headers:
                writer.writeheader()

            writer.writerow(data)
    except Exception as e:
        logger.error(f"Failed to write CSV log to {csv_path}: {e}")


def format_timestamp(ts: Optional[float] = None) -> str:
    """
    Format timestamp as ISO string.

    Args:
        ts: Unix timestamp. If None, uses current time.

    Returns:
        Formatted timestamp string
    """
    if ts is None:
        ts = time.time()
    return datetime.fromtimestamp(ts).isoformat()


def validate_percentage(value: float, name: str = "value") -> float:
    """
    Validate that a value is a valid percentage (0-100).

    Args:
        value: Value to validate
        name: Name of the value for error messages

    Returns:
        Validated percentage value

    Raises:
        ValueError: If value is not in valid range
    """
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {type(value)}")

    if not 0 <= value <= 100:
        raise ValueError(f"{name} must be between 0 and 100, got {value}")

    return float(value)


def validate_positive_number(value: float, name: str = "value") -> float:
    """
    Validate that a value is a positive number.

    Args:
        value: Value to validate
        name: Name of the value for error messages

    Returns:
        Validated positive value

    Raises:
        ValueError: If value is not positive
    """
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {type(value)}")

    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")

    return float(value)


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert value to float with default fallback.

    Args:
        value: Value to convert
        default: Default value if conversion fails

    Returns:
        Float value or default
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Clamp value between min and max.

    Args:
        value: Value to clamp
        min_val: Minimum value
        max_val: Maximum value

    Returns:
        Clamped value
    """
    return max(min_val, min(max_val, value))


def validate_user_requirements(requirements: Dict[str, Any]) -> Dict[str, float]:
    """
    Validate user requirements for accuracy, latency, and frequency.

    Args:
        requirements: Dictionary containing user requirements

    Returns:
        Validated requirements dictionary

    Raises:
        ValueError: If any requirement is invalid
    """
    validated = {}

    # Validate accuracy requirement (0-100%)
    if "min_accuracy" in requirements:
        validated["min_accuracy"] = validate_percentage(
            requirements["min_accuracy"], "min_accuracy"
        )
    else:
        raise ValueError("Missing required 'min_accuracy' in user requirements")

    # Validate latency requirement (positive ms)
    if "max_latency_ms" in requirements:
        validated["max_latency_ms"] = validate_positive_number(
            requirements["max_latency_ms"], "max_latency_ms"
        )
    else:
        raise ValueError("Missing required 'max_latency_ms' in user requirements")

    # Validate frequency requirement (positive ms)
    if "run_frequency_ms" in requirements:
        validated["run_frequency_ms"] = validate_positive_number(
            requirements["run_frequency_ms"], "run_frequency_ms"
        )
    else:
        raise ValueError("Missing required 'run_frequency_ms' in user requirements")

    return validated


def setup_logging(verbose: bool = False) -> None:
    """
    Setup logging configuration.

    Args:
        verbose: Enable verbose logging
    """
    import sys
    from loguru import logger

    # Remove default handler
    logger.remove()

    # Add console handler
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )


class Timer:
    """Simple timer context manager for measuring execution time."""

    def __init__(self, name: str = "operation"):
        """
        Initialize timer.

        Args:
            name: Name of the operation being timed
        """
        self.name = name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def __enter__(self):
        """Start timer."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timer."""
        self.end_time = time.perf_counter()
        if self.start_time is not None:
            duration = self.end_time - self.start_time
            logger.debug(f"{self.name} took {duration:.3f} seconds")

    @property
    def duration_ms(self) -> float:
        """Get duration in milliseconds."""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000
