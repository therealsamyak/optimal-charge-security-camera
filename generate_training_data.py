#!/usr/bin/env python3
"""
MIPS solver to generate training data for CustomController.
Generates optimal decisions using real energy data from all 4 locations.
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pulp
import concurrent.futures

# Add src directory to path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.power_profiler import PowerProfiler
from src.logging_config import setup_logging, get_logger

# Initialize logging
logger = setup_logging()


def load_energy_data() -> Dict[str, pd.DataFrame]:
    """Load energy data from all 4 locations."""
    logger.info("Starting energy data loading process")
    energy_dir = Path("energy-data")
    energy_data = {}

    # Find all CSV files in energy-data directory
    csv_files = list(energy_dir.glob("*.csv"))
    logger.info(f"Found {len(csv_files)} CSV files in {energy_dir}")

    for csv_file in csv_files:
        logger.debug(f"Processing energy file: {csv_file}")
        try:
            logger.debug(f"Reading CSV file: {csv_file}")
            df = pd.read_csv(csv_file, dtype=str, low_memory=False)
            logger.debug(f"Converting datetime column for {csv_file}")
            df["Datetime (UTC)"] = pd.to_datetime(df["Datetime (UTC)"])
            # Use filename without extension as location key
            location = csv_file.stem
            energy_data[location] = df
            logger.info(f"✓ Loaded {len(df)} records for {location}")
            logger.debug(
                f"Date range for {location}: {df['Datetime (UTC)'].min()} to {df['Datetime (UTC)'].max()}"
            )
        except Exception as e:
            logger.error(f"✗ Failed to load {csv_file}: {e}")
            logger.exception(f"Full traceback for {csv_file} loading error")

    logger.info(f"Energy data loading complete. Loaded {len(energy_data)} locations")
    return energy_data


def get_clean_energy_percentage(
    energy_data: Dict[str, pd.DataFrame], location: str, timestamp: datetime
) -> float:
    """Get clean energy percentage for specific location and timestamp."""
    logger.debug(f"Getting clean energy percentage for {location} at {timestamp}")

    if location not in energy_data:
        logger.warning(
            f"Location {location} not found in energy data, using fallback 50.0"
        )
        return 50.0  # Default fallback

    df = energy_data[location]
    logger.debug(f"Found {len(df)} records for {location}")

    # Find closest data point
    past_data = df[df["Datetime (UTC)"] <= timestamp]
    if past_data.empty:
        logger.debug(
            f"No past data found for {location} at {timestamp}, using fallback 50.0"
        )
        return 50.0

    latest_row = past_data.iloc[-1]
    cfe_col = "Carbon-free energy percentage (CFE%)"

    if cfe_col in latest_row:
        try:
            cfe_value = float(latest_row[cfe_col])
            logger.debug(
                f"Found exact CFE value for {location} at {timestamp}: {cfe_value}%"
            )
            return cfe_value
        except (ValueError, TypeError) as e:
            logger.debug(f"Failed to convert CFE value for {location}: {e}")

    # Handle missing data by averaging between before/after
    future_data = df[df["Datetime (UTC)"] > timestamp]
    if not future_data.empty:
        next_row = future_data.iloc[0]
        if cfe_col in next_row:
            try:
                before_val = float(latest_row.get(cfe_col, 50))
                after_val = float(next_row[cfe_col])
                avg_val = (before_val + after_val) / 2.0
                logger.debug(
                    f"Interpolated CFE value for {location} at {timestamp}: {avg_val}% (before: {before_val}%, after: {after_val}%)"
                )
                return avg_val
            except (ValueError, TypeError) as e:
                logger.debug(f"Failed to interpolate CFE value for {location}: {e}")

    logger.debug(
        f"No CFE data found for {location} at {timestamp}, using fallback 50.0"
    )
    return 50.0  # Default fallback


def load_power_profiles() -> Dict[str, Dict[str, float]]:
    """Load power profiles using PowerProfiler."""
    logger.info("Loading power profiles")
    from pathlib import Path
    import sys

    sys.path.insert(0, str(Path(__file__).parent / "src"))

    try:
        profiler = PowerProfiler()
        logger.debug("PowerProfiler instance created")

        profiler.load_profiles()  # Load profiles from file
        logger.debug("Power profiles loaded from file")

        # Transform profiles to match expected field names
        raw_profiles = profiler.get_all_models_data()
        logger.info(f"Found {len(raw_profiles)} raw model profiles")

        transformed_profiles = {}

        for name, profile in raw_profiles.items():
            logger.debug(f"Processing model profile: {name}")
            transformed_profiles[name] = {
                "accuracy": profile["accuracy"],
                "latency": profile["avg_inference_time_seconds"]
                * 1000,  # Convert to ms
                "power_cost": profile["model_power_mw"],  # Power in mW
            }
            logger.debug(
                f"Transformed {name}: accuracy={profile['accuracy']:.3f}, latency={transformed_profiles[name]['latency']:.2f}ms, power={profile['model_power_mw']:.2f}mW"
            )

        logger.info(
            f"✓ Successfully transformed {len(transformed_profiles)} model profiles"
        )
        return transformed_profiles

    except Exception as e:
        logger.error(f"✗ Failed to load power profiles: {e}")
        logger.exception("Full traceback for power profile loading error")
        raise


def solve_mips_scenario(
    battery_level: float,
    clean_energy_percentage: float,
    accuracy_requirement: float,
    latency_requirement: float,
    available_models: Dict[str, Dict[str, float]],
) -> Tuple[str, bool]:
    """
    Solve MIPS for a single scenario to get optimal model and charging decision.
    Objective: Maximize clean energy usage.
    """
    logger.debug(
        f"Solving MIPS scenario: battery={battery_level:.1f}%, clean_energy={clean_energy_percentage:.1f}%, acc_req={accuracy_requirement:.3f}, lat_req={latency_requirement}ms"
    )

    prob = pulp.LpProblem("Training_Scenario", pulp.LpMaximize)

    model_vars = {
        name: pulp.LpVariable(f"use_{name}", cat="Binary")
        for name in available_models.keys()
    }
    charge_var = pulp.LpVariable("charge", cat="Binary")
    logger.debug(f"Created {len(model_vars)} model variables and charge variable")

    # Objective: Maximize clean energy usage
    # Focus on charging when clean energy is high
    prob += clean_energy_percentage * charge_var
    logger.debug(f"Objective: maximize {clean_energy_percentage:.1f} * charge")

    # Constraint: Select exactly one model
    prob += pulp.lpSum(model_vars.values()) == 1
    logger.debug("Added constraint: exactly one model must be selected")

    # Filter models based on requirements
    filtered_models = []
    for name, specs in available_models.items():
        if specs["accuracy"] < accuracy_requirement:
            prob += model_vars[name] == 0
            logger.debug(
                f"Filtered out {name}: accuracy {specs['accuracy']:.3f} < requirement {accuracy_requirement:.3f}"
            )
        elif specs["latency"] > latency_requirement:
            prob += model_vars[name] == 0
            logger.debug(
                f"Filtered out {name}: latency {specs['latency']:.2f}ms > requirement {latency_requirement}ms"
            )
        else:
            filtered_models.append(name)
            logger.debug(
                f"Model {name} meets requirements: accuracy={specs['accuracy']:.3f}, latency={specs['latency']:.2f}ms"
            )

    # Battery capacity constraint
    prob += battery_level + charge_var * 15 <= 100
    logger.debug(f"Added battery constraint: {battery_level:.1f} + charge * 15 <= 100")

    logger.debug(f"Starting MIPS solver with {len(filtered_models)} eligible models")
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    status = pulp.LpStatus[prob.status]
    logger.debug(f"MIPS solver status: {status}")

    selected_model = list(available_models.keys())[0]
    for name, var in model_vars.items():
        if pulp.value(var) == 1:
            selected_model = name
            break

    should_charge = pulp.value(charge_var) == 1

    logger.debug(
        f"MIPS solution: selected_model={selected_model}, should_charge={should_charge}"
    )
    return selected_model, should_charge


def solve_scenario_wrapper(
    scenario_data: Tuple[int, int, float, int, str, datetime],
) -> Optional[Dict]:
    """Wrapper function for parallel execution of scenario solving."""
    battery, clean_energy, acc_req, lat_req, location, timestamp = scenario_data

    # Note: Logger won't work in worker processes, but we'll keep the structure
    # print statements are used for worker process debugging

    # Load data inside worker process
    models = load_power_profiles()

    try:
        selected_model, should_charge = solve_mips_scenario(
            battery, clean_energy, acc_req, lat_req, models
        )

        return {
            "battery_level": int(battery),
            "clean_energy_percentage": int(clean_energy),
            "accuracy_requirement": float(acc_req),
            "latency_requirement": int(lat_req),
            "location": location,
            "timestamp": timestamp.isoformat(),
            "optimal_model": selected_model,
            "should_charge": bool(should_charge),
        }
    except Exception as e:
        print(f"Error solving scenario: {e}")
        return None


def generate_training_scenarios(
    energy_data: Dict[str, pd.DataFrame],
    seed: Optional[int] = None,
) -> List[Tuple[int, int, float, int, str, datetime]]:
    """Generate realistic training scenarios using real energy data."""
    logger.info("Starting training scenario generation")

    if seed is not None:
        logger.info(f"Setting random seed: {seed}")
        random.seed(seed)
        np.random.seed(seed)

    locations = list(energy_data.keys())
    scenarios = []
    total_scenarios = 100000

    logger.info(
        f"Generating {total_scenarios:,} training scenarios across {len(locations)} locations"
    )

    # Track scenario distribution
    location_counts = {loc: 0 for loc in locations}
    clean_energy_values = []

    for i in range(total_scenarios):
        if i % 20000 == 0:
            progress = (i / total_scenarios) * 100
            logger.info(
                f"Scenario generation progress: {i:,}/{total_scenarios:,} ({progress:.1f}%)"
            )

        # Random location for maximal coverage
        location = random.choice(locations)
        location_counts[location] += 1

        # Random timestamp throughout the year
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 12, 31)
        random_timestamp = start_date + timedelta(
            seconds=random.randint(0, int((end_date - start_date).total_seconds()))
        )

        # Get real clean energy percentage
        clean_energy = get_clean_energy_percentage(
            energy_data, location, random_timestamp
        )
        clean_energy_values.append(clean_energy)

        # Debug progress every 1000 scenarios
        if i % 1000 == 0 and i > 0:
            progress = (i / total_scenarios) * 100
            logger.debug(
                f"Scenario generation progress: {i:,}/{total_scenarios:,} ({progress:.1f}%)"
            )

        # Random other parameters
        battery = np.random.uniform(1, 100)
        acc_req = np.random.uniform(0.3, 1.0)
        lat_req = np.random.choice([1, 2, 3, 5, 8, 10, 15, 20, 25, 30])

        scenarios.append(
            (battery, clean_energy, acc_req, lat_req, location, random_timestamp)
        )

    # Log generation statistics
    logger.info(f"✓ Generated {len(scenarios):,} training scenarios")
    logger.info("Scenario distribution by location:")
    for loc, count in location_counts.items():
        percentage = (count / total_scenarios) * 100
        logger.info(f"  {loc}: {count:,} scenarios ({percentage:.1f}%)")

    if clean_energy_values:
        avg_clean_energy = np.mean(clean_energy_values)
        logger.info(
            f"Average clean energy percentage across all scenarios: {avg_clean_energy:.1f}%"
        )

    return scenarios


def main():
    """Generate training data and save to JSON."""
    logger.info("=" * 80)
    logger.info("STARTING TRAINING DATA GENERATION")
    logger.info("=" * 80)

    start_time = datetime.now()
    logger.info(f"Start time: {start_time}")

    # Ensure results directory exists
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    logger.info(f"Results directory: {results_dir}")

    print("Loading energy data...")
    logger.info("Loading energy data...")
    try:
        energy_data = load_energy_data()
        print(f"✓ Loaded energy data for {len(energy_data)} regions")
        logger.info(f"✓ Loaded energy data for {len(energy_data)} regions")
    except Exception as e:
        print(f"✗ Error loading energy data: {e}")
        logger.error(f"✗ Error loading energy data: {e}")
        logger.exception("Full traceback for energy data loading error")
        return

    print("Loading power profiles...")
    logger.info("Loading power profiles...")
    try:
        models = load_power_profiles()
        print(f"✓ Loaded {len(models)} model profiles")
        logger.info(f"✓ Loaded {len(models)} model profiles")
        if not models:
            print("✗ No model profiles found - power_profiles.json may be empty")
            logger.error("✗ No model profiles found - power_profiles.json may be empty")
            return
    except Exception as e:
        print(f"✗ Error loading power profiles: {e}")
        logger.error(f"✗ Error loading power profiles: {e}")
        logger.exception("Full traceback for power profiles loading error")
        return

    print("Generating training scenarios with real energy data...")
    logger.info("Generating training scenarios with real energy data...")
    try:
        scenarios = generate_training_scenarios(energy_data)
        print(f"✓ Generated {len(scenarios)} training scenarios")
        logger.info(f"✓ Generated {len(scenarios)} training scenarios")
    except Exception as e:
        print(f"✗ Error generating scenarios: {e}")
        logger.error(f"✗ Error generating scenarios: {e}")
        logger.exception("Full traceback for scenario generation error")
        return

    print(f"Solving MIPS for {len(scenarios)} scenarios in parallel...")
    logger.info(f"Solving MIPS for {len(scenarios)} scenarios in parallel...")
    training_data = []
    # Use ProcessPoolExecutor for parallel execution
    max_workers = 100
    print(f"Starting parallel processing with {max_workers} workers...")
    logger.info(f"Starting parallel processing with {max_workers} workers...")
    print(f"Submitting {len(scenarios)} scenarios to worker pool...")
    logger.info(f"Submitting {len(scenarios)} scenarios to worker pool...")

    try:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
            # Submit all scenarios for processing
            future_to_scenario = {
                executor.submit(solve_scenario_wrapper, scenario): i
                for i, scenario in enumerate(scenarios)
            }
            print(f"✓ All {len(scenarios)} scenarios submitted to workers")
            logger.info(f"✓ All {len(scenarios)} scenarios submitted to workers")

            completed_count = 0
            error_count = 0
            # Collect results as they complete
            print("Waiting for scenarios to complete...")
            logger.info("Waiting for scenarios to complete...")
            for future in concurrent.futures.as_completed(future_to_scenario):
                scenario_index = future_to_scenario[future]
                completed_count += 1

                # Progress updates more frequently for better visibility
                if completed_count % 100 == 0:
                    progress_pct = (completed_count / len(scenarios)) * 100
                    print(
                        f"Progress: {completed_count}/{len(scenarios)} ({progress_pct:.1f}%) - Errors: {error_count}"
                    )
                    logger.info(
                        f"Progress: {completed_count:,}/{len(scenarios):,} ({progress_pct:.1f}%) - Errors: {error_count}"
                    )
                elif completed_count % 25 == 0:
                    progress_pct = (completed_count / len(scenarios)) * 100
                    print(
                        f"Progress: {completed_count}/{len(scenarios)} ({progress_pct:.1f}%)"
                    )
                    logger.debug(
                        f"Progress: {completed_count:,}/{len(scenarios):,} ({progress_pct:.1f}%)"
                    )

                try:
                    result = future.result()
                    if result:
                        training_data.append(result)
                    else:
                        error_count += 1
                except Exception as e:
                    error_count += 1
                    if error_count <= 5:  # Show first 5 errors to avoid spam
                        print(f"Error processing scenario {scenario_index}: {e}")
                        logger.error(f"Error processing scenario {scenario_index}: {e}")
                    continue

        success_rate = (len(training_data) / len(scenarios)) * 100
        print(
            f"✓ Parallel processing completed - {len(training_data)} successful, {error_count} failed ({success_rate:.1f}% success rate)"
        )
        logger.info(
            f"✓ Parallel processing completed - {len(training_data)} successful, {error_count} failed ({success_rate:.1f}% success rate)"
        )
    except Exception as e:
        print(f"✗ Error in parallel processing: {e}")
        logger.error(f"✗ Error in parallel processing: {e}")
        logger.exception("Full traceback for parallel processing error")
        return

    print(f"✓ Generated {len(training_data)} training samples")
    logger.info(f"✓ Generated {len(training_data)} training samples")

    # Save to JSON
    output_file = "results/training_data.json"
    print(f"Saving training data to {output_file}...")
    logger.info(f"Saving training data to {output_file}...")
    try:
        # Create results directory if it doesn't exist
        results_dir = Path(output_file).parent
        results_dir.mkdir(exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(training_data, f, indent=2)
        print("✓ Training data saved to results/training_data.json")
        logger.info(f"✓ Training data saved to {output_file}")

        # Log file size
        file_size = Path(output_file).stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        logger.info(f"Training data file size: {file_size_mb:.2f} MB")

    except Exception as e:
        print(f"✗ Error saving training data: {e}")
        logger.error(f"✗ Error saving training data: {e}")
        logger.exception("Full traceback for saving training data error")
        return

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"✓ Training data generation completed in {duration:.2f} seconds")
    logger.info(f"✓ Training data generation completed in {duration:.2f} seconds")
    logger.info("=" * 80)
    logger.info("TRAINING DATA GENERATION COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
