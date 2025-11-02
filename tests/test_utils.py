"""
Unit tests for utility functions.

Tests helper functions and utilities.
"""

import pytest
import tempfile
import csv
import time
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils.helpers import (
    ensure_directory,
    write_csv_log,
    format_timestamp,
    validate_percentage,
    validate_positive_number,
    safe_float,
    clamp,
    Timer,
    validate_user_requirements,
)


class TestEnsureDirectory:
    """Test cases for ensure_directory function."""

    def test_create_new_directory(self):
        """Test creating a new directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            new_dir = os.path.join(temp_dir, "new", "nested", "directory")

            # Directory should not exist initially
            assert not os.path.exists(new_dir)

            # Create directory
            ensure_directory(new_dir)

            # Directory should now exist
            assert os.path.exists(new_dir)
            assert os.path.isdir(new_dir)

    def test_existing_directory(self):
        """Test handling of existing directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Directory already exists
            ensure_directory(temp_dir)

            # Should not raise an error
            assert os.path.exists(temp_dir)


class TestWriteCsvLog:
    """Test cases for write_csv_log function."""

    def test_write_new_csv(self):
        """Test writing to a new CSV file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            csv_path = f.name

        try:
            data = {
                "timestamp": "2025-01-01T12:00:00",
                "model": "yolov10n",
                "confidence": 0.85,
                "latency_ms": 25.5,
            }

            write_csv_log(csv_path, data)

            # Read back and verify
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 1
            assert rows[0]["timestamp"] == "2025-01-01T12:00:00"
            assert rows[0]["model"] == "yolov10n"
            assert rows[0]["confidence"] == "0.85"
            assert rows[0]["latency_ms"] == "25.5"

        finally:
            os.unlink(csv_path)

    def test_append_to_existing_csv(self):
        """Test appending to an existing CSV file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            csv_path = f.name

        try:
            # Write first entry
            data1 = {"model": "yolov10n", "confidence": 0.85}
            write_csv_log(csv_path, data1)

            # Write second entry
            data2 = {"model": "yolov10s", "confidence": 0.90}
            write_csv_log(csv_path, data2)

            # Read back and verify
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 2
            assert rows[0]["model"] == "yolov10n"
            assert rows[1]["model"] == "yolov10s"

        finally:
            os.unlink(csv_path)


class TestFormatTimestamp:
    """Test cases for format_timestamp function."""

    def test_format_current_time(self):
        """Test formatting current time."""
        timestamp_str = format_timestamp()

        # Should be in ISO format
        assert "T" in timestamp_str
        assert len(timestamp_str) > 15

    def test_format_specific_time(self):
        """Test formatting specific timestamp."""
        test_time = 1640995200.0  # 2022-01-01 00:00:00 UTC
        timestamp_str = format_timestamp(test_time)

        # Should contain expected date
        assert "2022-01-01" in timestamp_str


class TestValidatePercentage:
    """Test cases for validate_percentage function."""

    def test_valid_percentages(self):
        """Test valid percentage values."""
        assert validate_percentage(0) == 0.0
        assert validate_percentage(50) == 50.0
        assert validate_percentage(100) == 100.0
        assert validate_percentage(25.5) == 25.5

    def test_invalid_percentages(self):
        """Test invalid percentage values."""
        with pytest.raises(ValueError, match="must be between 0 and 100"):
            validate_percentage(-1)

        with pytest.raises(ValueError, match="must be between 0 and 100"):
            validate_percentage(101)

        with pytest.raises(ValueError, match="must be a number"):
            validate_percentage("not_a_number")

    def test_custom_name(self):
        """Test custom error message name."""
        with pytest.raises(ValueError, match="custom_value must be between 0 and 100"):
            validate_percentage(150, "custom_value")


class TestValidatePositiveNumber:
    """Test cases for validate_positive_number function."""

    def test_valid_numbers(self):
        """Test valid positive numbers."""
        assert validate_positive_number(1) == 1.0
        assert validate_positive_number(50.5) == 50.5
        assert validate_positive_number(0.1) == 0.1

    def test_invalid_numbers(self):
        """Test invalid positive numbers."""
        with pytest.raises(ValueError, match="must be positive"):
            validate_positive_number(0)

        with pytest.raises(ValueError, match="must be positive"):
            validate_positive_number(-1)

        with pytest.raises(ValueError, match="must be a number"):
            validate_positive_number("not_a_number")

    def test_custom_name(self):
        """Test custom error message name."""
        with pytest.raises(ValueError, match="custom_value must be positive"):
            validate_positive_number(0, "custom_value")


class TestSafeFloat:
    """Test cases for safe_float function."""

    def test_valid_conversions(self):
        """Test valid float conversions."""
        assert safe_float("123.45") == 123.45
        assert safe_float(42) == 42.0
        assert safe_float(3.14) == 3.14

    def test_invalid_conversions(self):
        """Test invalid float conversions."""
        assert safe_float("not_a_number") == 0.0
        assert safe_float(None) == 0.0
        assert safe_float([]) == 0.0

    def test_custom_default(self):
        """Test custom default value."""
        assert safe_float("invalid", 99.9) == 99.9
        assert safe_float(None, -1.0) == -1.0


class TestClamp:
    """Test cases for clamp function."""

    def test_within_range(self):
        """Test values within range."""
        assert clamp(50, 0, 100) == 50
        assert clamp(25, 10, 30) == 25

    def test_below_range(self):
        """Test values below range."""
        assert clamp(-5, 0, 100) == 0
        assert clamp(5, 10, 30) == 10

    def test_above_range(self):
        """Test values above range."""
        assert clamp(150, 0, 100) == 100
        assert clamp(35, 10, 30) == 30


class TestTimer:
    """Test cases for Timer class."""

    def test_timer_context_manager(self):
        """Test timer as context manager."""
        with Timer("test_operation") as timer:
            time.sleep(0.1)  # Sleep for 100ms

        # Duration should be positive
        assert timer.duration_ms > 50  # Allow some tolerance
        assert timer.duration_ms < 200  # Should be close to 100ms

    def test_timer_properties(self):
        """Test timer properties."""
        timer = Timer("test")

        # Before context, duration should be 0
        assert timer.duration_ms == 0.0

        with timer:
            time.sleep(0.05)

        # After context, duration should be positive
        assert timer.duration_ms > 30


class TestValidateUserRequirements:
    """Test cases for validate_user_requirements function."""

    def test_valid_requirements(self):
        """Test valid user requirements."""
        requirements = {
            "min_accuracy": 85.0,
            "max_latency_ms": 75.0,
            "run_frequency_ms": 1500.0,
        }

        validated = validate_user_requirements(requirements)

        assert validated["min_accuracy"] == 85.0
        assert validated["max_latency_ms"] == 75.0
        assert validated["run_frequency_ms"] == 1500.0

    def test_invalid_accuracy(self):
        """Test invalid accuracy requirement."""
        requirements = {
            "min_accuracy": 150.0,  # Invalid
            "max_latency_ms": 75.0,
            "run_frequency_ms": 1500.0,
        }

        with pytest.raises(ValueError, match="min_accuracy must be between 0 and 100"):
            validate_user_requirements(requirements)

    def test_invalid_latency(self):
        """Test invalid latency requirement."""
        requirements = {
            "min_accuracy": 85.0,
            "max_latency_ms": -10.0,  # Invalid
            "run_frequency_ms": 1500.0,
        }

        with pytest.raises(ValueError, match="max_latency_ms must be positive"):
            validate_user_requirements(requirements)

    def test_invalid_frequency(self):
        """Test invalid frequency requirement."""
        requirements = {
            "min_accuracy": 85.0,
            "max_latency_ms": 75.0,
            "run_frequency_ms": 0.0,  # Invalid
        }

        with pytest.raises(ValueError, match="run_frequency_ms must be positive"):
            validate_user_requirements(requirements)

    def test_missing_requirements(self):
        """Test missing required fields."""
        # Missing min_accuracy
        requirements = {"max_latency_ms": 75.0, "run_frequency_ms": 1500.0}

        with pytest.raises(ValueError, match="Missing required 'min_accuracy'"):
            validate_user_requirements(requirements)

        # Missing max_latency_ms
        requirements = {"min_accuracy": 85.0, "run_frequency_ms": 1500.0}

        with pytest.raises(ValueError, match="Missing required 'max_latency_ms'"):
            validate_user_requirements(requirements)

        # Missing run_frequency_ms
        requirements = {"min_accuracy": 85.0, "max_latency_ms": 75.0}

        with pytest.raises(ValueError, match="Missing required 'run_frequency_ms'"):
            validate_user_requirements(requirements)
