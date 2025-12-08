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
    from src.full_horizon_training import generate_full_horizon_training_data
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
        # Generate training data using full-horizon MILP optimization
        training_data = generate_full_horizon_training_data()

        print(f"✓ Generated {len(training_data)} training samples")

        # Save to JSON file
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        output_file = results_dir / "training_data.json"

        with open(output_file, "w") as f:
            json.dump(training_data, f, indent=2)

        print(f"✓ Training data saved to {output_file}")

    except Exception as e:
        print(f"✗ Error generating training data: {e}")
        return

    print("✓ Training data generation completed successfully")


if __name__ == "__main__":
    main()
