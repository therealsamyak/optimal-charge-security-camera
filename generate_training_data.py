#!/usr/bin/env python3
"""
Generate training data using full-horizon optimization.
Simple wrapper that calls existing full-horizon training implementation.
"""

import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from full_horizon_training import generate_full_horizon_training_data
except ImportError as e:
    print(f"Import error: {e}")
    print(
        "Make sure you're running this from project root with 'uv run generate_training_data.py'"
    )
    sys.exit(1)


def main():
    """Generate training data and save to JSON."""
    print("Generating full-horizon training data...")

    try:
        # Generate training data using existing full-horizon implementation
        training_data = generate_full_horizon_training_data()

        # Save to JSON
        output_file = "results/training_data.json"
        print(f"Saving training data to {output_file}...")

        with open(output_file, "w") as f:
            json.dump(training_data, f, indent=2)

        print(f"✓ Generated {len(training_data)} training samples")
        print(f"✓ Training data saved to results/training_data.json")

        # Log file size
        file_size = Path(output_file).stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        print(f"File size: {file_size_mb:.2f} MB")

    except Exception as e:
        print(f"✗ Error generating training data: {e}")
        return

    print("✓ Training data generation completed successfully")


if __name__ == "__main__":
    main()
