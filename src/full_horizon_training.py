"""
Full-horizon training data generation using optimal MILP solutions.

This module generates training data by solving full-horizon MILP optimization
problems for representative days across different locations and seasons.
The resulting training data contains truly optimal decisions for both model selection
and charging decisions.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import pulp

from config_loader import ConfigLoader, SimulationConfig
from energy_data import EnergyData

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)  # Reduce logging verbosity


def solve_full_horizon_milp(
    clean_energy_series: List[float],
    task_requirements: List[Dict],
    config: SimulationConfig,
) -> List[Tuple[str, bool]]:
    """
    Shared MILP solver for both oracle and training data.

    Args:
        clean_energy_series: Clean energy percentages for each timestep
        task_requirements: Task requirements for each timestep
        config: Simulation configuration

    Returns:
        List of (model_name, should_charge) tuples for each timestep
    """
    # Load model data from JSON file
    import json
    from pathlib import Path

    profiles_file = Path("model-data/power_profiles.json")
    with open(profiles_file, "r") as f:
        raw_models = json.load(f)

    # Transform profiles to match expected field names
    available_models = {}
    for name, profile in raw_models.items():
        available_models[name] = {
            "accuracy": profile["accuracy"],
            "latency": profile["avg_inference_time_seconds"] * 1000,  # Convert to ms
            "power_cost": profile["model_power_mw"],  # Power in mW
        }

    # Use runtime config parameters
    task_interval = config.task_interval_seconds  # 5s
    battery_capacity = config.battery_capacity_wh  # 5.0Wh
    charge_rate = config.charge_rate_watts  # 100W

    # Calculate number of timesteps
    num_timesteps = len(clean_energy_series)

    prob = pulp.LpProblem("Full_Horizon_Training_Generation", pulp.LpMaximize)

    # Decision variables for all timesteps
    model_vars = {}
    charge_vars = {}
    battery_vars = {}

    for t in range(num_timesteps):
        model_vars[t] = {
            name: pulp.LpVariable(f"model_{t}_{name}", cat="Binary")
            for name in available_models.keys()
        }
        charge_vars[t] = pulp.LpVariable(f"charge_{t}", cat="Binary")
        battery_vars[t] = pulp.LpVariable(f"battery_{t}", lowBound=0, upBound=100)

    # Objective: maximize clean energy usage
    prob += pulp.lpSum(
        [clean_energy_series[t] * charge_vars[t] for t in range(num_timesteps)]
    )

    # Constraints
    for t in range(num_timesteps):
        # Exactly one model per timestep
        prob += pulp.lpSum(model_vars[t].values()) == 1

        # Task requirements
        task_req = task_requirements[t]
        for name, specs in available_models.items():
            if specs["accuracy"] < task_req["accuracy"]:
                prob += model_vars[t][name] == 0
            if specs["latency"] > task_req["latency"]:
                prob += model_vars[t][name] == 0

        # Battery dynamics
        if t == 0:
            # Initial battery level (start at 50%)
            prob += battery_vars[t] == 50
        else:
            # Battery transition: battery_t = battery_{t-1} + charge_{t-1} * rate - energy_used_{t-1}

            # Energy consumed in mWh for the task interval
            energy_used_mwh = pulp.lpSum(
                [
                    available_models[name]["power_cost"]
                    * model_vars[t - 1][name]
                    * task_interval
                    / 3600
                    for name in available_models.keys()
                ]
            )

            # Convert mWh to battery percentage
            energy_used_percent = energy_used_mwh / battery_capacity * 100

            # Charging: convert watts to mWh, then to percentage
            charge_added_mwh = (
                charge_vars[t - 1] * charge_rate * task_interval / 1000
            )  # W to mWh
            charge_added_percent = charge_added_mwh / battery_capacity * 100

            prob += (
                battery_vars[t]
                == battery_vars[t - 1] + charge_added_percent - energy_used_percent
            )

        # Battery bounds
        prob += battery_vars[t] >= 0
        prob += battery_vars[t] <= 100

    # Solve MILP
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # Extract optimal schedule
    optimal_schedule = []
    for t in range(num_timesteps):
        selected_model = None
        for name, var in model_vars[t].items():
            if pulp.value(var) == 1:
                selected_model = name
                break

        should_charge = pulp.value(charge_vars[t]) == 1
        optimal_schedule.append((selected_model, should_charge))

    return optimal_schedule


def get_representative_day_for_season(season: str) -> Dict[str, int]:
    """Get representative day for season"""
    season_days = {
        "winter": {"month": 1, "day": 15},  # January 15
        "spring": {"month": 4, "day": 15},  # April 15
        "summer": {"month": 7, "day": 15},  # July 15
        "fall": {"month": 10, "day": 15},  # October 15
    }
    return season_days.get(season, {"month": 1, "day": 15})


def get_location_filename(location: str) -> str:
    """Get energy data filename for location"""
    location_mapping = {
        "CA": "US-CAL-LDWP_2024_5_minute",
        "FL": "US-FLA-FPL_2024_5_minute",
        "NW": "US-NW-PSEI_2024_5_minute",
        "NY": "US-NY-NYIS_2024_5_minute",
    }
    return location_mapping.get(location, "US-CAL-LDWP_2024_5_minute")


def get_clean_energy_series(
    location: str, season: str, config: SimulationConfig
) -> List[float]:
    """Get clean energy percentages for full day"""
    energy_data = EnergyData()
    day = get_representative_day_for_season(season)
    location_file = get_location_filename(location)

    clean_energy_series = []
    duration_seconds = config.duration_days * 24 * 3600
    task_interval = config.task_interval_seconds

    for timestamp in range(0, int(duration_seconds), task_interval):
        dt = datetime(2024, day["month"], day["day"]) + timedelta(seconds=timestamp)
        clean_energy_pct = energy_data.get_clean_energy_percentage(
            location_file, dt.strftime("%Y-%m-%d %H:%M:%S")
        )
        clean_energy_series.append(clean_energy_pct or 0.0)

    return clean_energy_series


def generate_task_requirements(config: SimulationConfig) -> List[Dict]:
    """Generate task requirements for entire day"""
    task_requirements = []
    duration_seconds = config.duration_days * 24 * 3600
    task_interval = config.task_interval_seconds

    for timestamp in range(0, int(duration_seconds), task_interval):
        task_requirements.append(
            {
                "accuracy": config.user_accuracy_requirement
                / 100.0,  # Convert to decimal
                "latency": config.user_latency_requirement,
            }
        )

    return task_requirements


def calculate_battery_at_timestep(
    t: int, optimal_schedule: List[Tuple[str, bool]], config: SimulationConfig
) -> float:
    """Calculate battery level at specific timestep"""
    # Load model data from JSON file
    import json
    from pathlib import Path

    profiles_file = Path("model-data/power_profiles.json")
    with open(profiles_file, "r") as f:
        raw_models = json.load(f)

    # Transform profiles
    available_models = {}
    for name, profile in raw_models.items():
        available_models[name] = {
            "power_cost": profile["model_power_mw"],  # Power in mW
        }

    battery_level = 50.0  # Start at 50%
    task_interval = config.task_interval_seconds
    battery_capacity = config.battery_capacity_wh
    charge_rate = config.charge_rate_watts

    for i in range(t):
        model_name, should_charge = optimal_schedule[i]

        # Energy consumed in mWh for the task interval
        energy_used_mwh = (
            available_models[model_name]["power_cost"] * task_interval / 3600
        )

        # Convert mWh to battery percentage
        energy_used_percent = energy_used_mwh / battery_capacity * 100
        battery_level -= energy_used_percent

        # Charge if needed
        if should_charge:
            # Charging: convert watts to mWh, then to percentage
            charge_added_mwh = charge_rate * task_interval / 1000  # W to mWh
            charge_added_percent = charge_added_mwh / battery_capacity * 100
            battery_level += charge_added_percent

        # Ensure bounds
        battery_level = max(0, min(100, battery_level))

    return battery_level


def generate_day_training_data(
    work_item: Tuple[str, str, str], config: SimulationConfig
) -> List[Dict]:
    """Generate training data for one day using full-horizon MILP"""
    location, season, day_str = work_item

    logger.info(f"Generating training data for {location} {season} {day_str}")

    # Extract full day's clean energy series
    clean_energy_series = get_clean_energy_series(location, season, config)

    # Generate task requirements for day
    task_requirements = generate_task_requirements(config)

    # Solve full-horizon MILP
    optimal_schedule = solve_full_horizon_milp(
        clean_energy_series, task_requirements, config
    )

    # Extract training examples from optimal schedule
    training_examples = []
    for t, (model, charge) in enumerate(optimal_schedule):
        example = {
            "battery_level": calculate_battery_at_timestep(t, optimal_schedule, config),
            "clean_energy_percentage": clean_energy_series[t],
            "accuracy_requirement": task_requirements[t]["accuracy"],
            "latency_requirement": task_requirements[t]["latency"],
            "optimal_model": model,
            "should_charge": charge,
        }
        training_examples.append(example)

    logger.info(
        f"Generated {len(training_examples)} training examples for {location} {season}"
    )
    return training_examples


def generate_full_horizon_training_data() -> List[Dict]:
    """Generate training data using full-horizon optimization with parallel processing."""
    logger.info("=" * 80)
    logger.info("STARTING FULL-HORIZON TRAINING DATA GENERATION")
    logger.info("=" * 80)

    # Load configuration
    config_loader = ConfigLoader()
    config = config_loader.get_simulation_config()
    workers_config = config_loader.get_workers_config()

    max_workers = workers_config.max_workers
    logger.info(f"Using {max_workers} workers for parallel processing")

    # Create work items: (location, season, day) combinations
    work_items = []
    for location in config.locations:  # ["CA", "FL", "NW", "NY"]
        for season in config.seasons:  # ["winter", "spring", "summer", "fall"]
            day = get_representative_day_for_season(season)
            day_str = f"{day['month']:02d}-{day['day']:02d}"
            work_items.append((location, season, day_str))

    logger.info(f"Processing {len(work_items)} location-season combinations")

    # Parallel processing using config workers
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(generate_day_training_data, item, config)
            for item in work_items
        ]

        # Collect results
        all_training_data = []
        for future in as_completed(futures):
            try:
                day_data = future.result()
                all_training_data.extend(day_data)
            except Exception as e:
                logger.error(f"Error processing training data: {e}")

    logger.info(f"Generated {len(all_training_data)} total training examples")
    return all_training_data


def main():
    """Generate training data and save to JSON."""
    import json

    # Ensure results directory exists
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    print("Generating full-horizon training data...")
    logger.info("Generating full-horizon training data...")

    try:
        training_data = generate_full_horizon_training_data()

        # Save to JSON file
        output_file = results_dir / "full_horizon_training_data.json"
        with open(output_file, "w") as f:
            json.dump(training_data, f, indent=2)

        print(f"✓ Saved {len(training_data)} training examples to {output_file}")
        logger.info(f"✓ Saved {len(training_data)} training examples to {output_file}")

    except Exception as e:
        print(f"✗ Error generating training data: {e}")
        logger.error(f"✗ Error generating training data: {e}")
        logger.exception("Full traceback for training data generation error")


if __name__ == "__main__":
    main()
