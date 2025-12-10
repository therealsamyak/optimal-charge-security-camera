#!/usr/bin/env python3
"""
Generate training data using full-horizon MILP optimization.

Creates training data for all combinations of:
- 4 locations (CA, FL, NW, NY)
- 4 seasons (winter, spring, summer, fall)
- Single accuracy/latency requirement from config

Total: 16 scenarios with optimal decisions for neural controller training.
"""

import json
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.full_horizon_training import generate_day_training_data
    from src.config_loader import ConfigLoader
except ImportError as e:
    print(f"Import error: {e}")
    print(
        "Make sure you're running this from project root with 'uv run generate_training_data.py'"
    )
    sys.exit(1)


# Use single hardcoded requirement scenario from config
def get_requirement_scenario():
    """Get single requirement scenario from config."""
    config_loader = ConfigLoader()
    config = config_loader.get_simulation_config()
    return {
        "accuracy": config.user_accuracy_requirement,
        "latency": config.user_latency_requirement,
    }


def generate_scenario_training_data(
    work_item: Tuple[str, str, str, Dict],
) -> List[Dict]:
    """Generate training data for one scenario with specific requirements."""
    location, season, requirement_level, requirements = work_item

    print(f"🌍 Generating training data for {location} {season}")
    print(
        f"📋 Requirements: {requirements['accuracy']}% accuracy, {requirements['latency']}s latency"
    )

    # Load config and update with scenario requirements
    config_loader = ConfigLoader()
    config = config_loader.get_simulation_config()
    config.user_accuracy_requirement = requirements["accuracy"]
    config.user_latency_requirement = requirements["latency"]

    # Generate day identifier
    from src.full_horizon_training import get_representative_day_for_season

    day = get_representative_day_for_season(season)
    day_str = f"{day['month']:02d}-{day['day']:02d}"

    # Generate training data using existing MILP solver
    day_data = generate_day_training_data((location, season, day_str), config)

    # Add scenario metadata to each training example
    for example in day_data:
        example["scenario_metadata"] = {
            "location": location,
            "season": season,
            "requirement_level": requirement_level,
        }

    print(f"✅ Generated {len(day_data)} examples for {location} {season}")
    return day_data


def main():
    """Generate comprehensive training data for all scenarios."""
    print("=" * 80)
    print("🚀 STARTING COMPREHENSIVE TRAINING DATA GENERATION")
    print("=" * 80)

    # Load configuration for parallel processing
    config_loader = ConfigLoader()
    config = config_loader.get_simulation_config()
    workers_config = config_loader.get_workers_config()
    max_workers = workers_config.max_workers

    print(f"🔧 Using {max_workers} workers for parallel processing")

    # Get single requirement scenario from config
    requirements = get_requirement_scenario()

    # Generate all work items: 4 locations × 4 seasons = 16 scenarios
    work_items = []
    for location in config.locations:  # ["CA", "FL", "NW", "NY"]
        for season in config.seasons:  # ["winter", "spring", "summer", "fall"]
            work_items.append((location, season, "config", requirements))

    print(f"📅 Processing {len(work_items)} total scenarios")
    print(f"🌟 Locations: {config.locations}")
    print(f"🌙 Seasons: {config.seasons}")
    print(
        f"📊 Single requirement: {requirements['accuracy']}% accuracy, {requirements['latency']}s latency"
    )

    # Parallel processing
    print("⚡ Starting parallel processing...")
    all_training_data = []
    completed = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(generate_scenario_training_data, item)
            for item in work_items
        ]

        for future in as_completed(futures):
            try:
                scenario_data = future.result()
                all_training_data.extend(scenario_data)
                completed += 1
                print(
                    f"✅ Completed {completed}/{len(work_items)} scenarios ({len(scenario_data)} examples)"
                )
            except Exception as e:
                print(f"❌ Error processing scenario: {e}")
                import traceback

                traceback.print_exc()

    print(f"🎉 Generated {len(all_training_data)} total training examples")

    # Save to JSON file
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    output_file = results_dir / "training_data.json"

    print(f"💾 Saving training data to {output_file}")
    with open(output_file, "w") as f:
        json.dump(all_training_data, f, indent=2)

    print(f"✓ Training data saved to {output_file}")

    # Print summary statistics
    print("\n📊 TRAINING DATA SUMMARY:")
    scenario_counts = {}
    for example in all_training_data:
        metadata = example["scenario_metadata"]
        key = f"{metadata['location']}_{metadata['season']}_{metadata['requirement_level']}"
        scenario_counts[key] = scenario_counts.get(key, 0) + 1

    print(f"   Total examples: {len(all_training_data)}")
    print(f"   Total scenarios: {len(scenario_counts)}")
    print(
        f"   Examples per scenario: ~{len(all_training_data) // len(scenario_counts)}"
    )

    print("\n✓ Training data generation completed successfully")


if __name__ == "__main__":
    main()
