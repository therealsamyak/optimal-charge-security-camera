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
) -> List[Tuple[str, bool, str]]:
    """
    Shared MILP solver for both oracle and training data.
    Handles infeasibility by allowing task misses and super misses.

    Args:
        clean_energy_series: Clean energy percentages for each timestep
        task_requirements: Task requirements for each timestep
        config: Simulation configuration

    Returns:
        List of (model_name, should_charge, status) tuples for each timestep
        status: "success", "small_miss", "large_miss"
    """
    print(f"🔧 Starting MILP solver with {len(clean_energy_series)} timesteps")
    print(
        f"📊 Battery: {config.battery_capacity_wh}Wh, Charge: {config.charge_rate_watts}W"
    )
    print(f"⚡ Task interval: {config.task_interval_seconds}s")
    print(
        f"🎯 Requirements: {config.user_accuracy_requirement}% accuracy, {config.user_latency_requirement}s latency"
    )
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
            "energy_per_task_mwh": profile[
                "energy_per_inference_mwh"
            ],  # Energy per task in mWh
        }

    # Use runtime config parameters
    task_interval = config.task_interval_seconds  # 5s
    battery_capacity = (
        config.battery_capacity_wh * 1000
    )  # Convert Wh to mWh (20.0Wh = 20000mWh)
    charge_rate = config.charge_rate_watts  # 100W

    # Calculate number of timesteps
    num_timesteps = len(clean_energy_series)

    prob = pulp.LpProblem("Full_Horizon_Training_Generation", pulp.LpMaximize)

    # Decision variables for all timesteps
    model_vars = {}
    charge_vars = {}
    battery_vars = {}
    small_miss_vars = {}
    large_miss_vars = {}
    super_miss_vars = {}

    for t in range(num_timesteps):
        model_vars[t] = {
            name: pulp.LpVariable(f"model_{t}_{name}", cat="Binary")
            for name in available_models.keys()
        }
        charge_vars[t] = pulp.LpVariable(f"charge_{t}", cat="Binary")
        battery_vars[t] = pulp.LpVariable(f"battery_{t}", lowBound=0, upBound=100)
        small_miss_vars[t] = pulp.LpVariable(f"small_miss_{t}", cat="Binary")
        large_miss_vars[t] = pulp.LpVariable(f"large_miss_{t}", cat="Binary")
        super_miss_vars[t] = pulp.LpVariable(f"super_miss_{t}", cat="Binary")

    # Objective: maximize successful tasks, penalize misses heavily
    # Priority: 1) Max successful tasks, 2) Minimize super misses, 3) Minimize task misses, 4) Use clean energy when possible
    successful_tasks = (
        pulp.lpSum(
            [
                model_vars[t][name]
                for t in range(num_timesteps)
                for name in available_models.keys()
            ]
        )
        * 10000
    )  # Very high weight for successful tasks
    clean_energy_reward = (
        pulp.lpSum(
            [clean_energy_series[t] * charge_vars[t] for t in range(num_timesteps)]
        )
        * 0.1  # Very small bonus for clean charging
    )
    small_miss_penalty = (
        pulp.lpSum([small_miss_vars[t] for t in range(num_timesteps)]) * 5000
    )
    super_miss_penalty = (
        pulp.lpSum([super_miss_vars[t] for t in range(num_timesteps)]) * 10000
    )

    # Add penalty for excessive charging
    excessive_charge_penalty = (
        pulp.lpSum([charge_vars[t] for t in range(num_timesteps)]) * 100
    )

    prob += (
        successful_tasks
        + clean_energy_reward
        - small_miss_penalty
        - super_miss_penalty
        - excessive_charge_penalty
    )

    # Constraints
    for t in range(num_timesteps):
        # Exactly one action per timestep: model OR charge OR miss
        prob += (
            pulp.lpSum(model_vars[t].values())
            + charge_vars[t]
            + small_miss_vars[t]
            + large_miss_vars[t]
            + super_miss_vars[t]
            == 1
        )

        # Task requirements: accuracy >= requirement, latency <= requirement
        task_req = task_requirements[t]
        eligible_models = []
        ineligible_models = []
        for name, specs in available_models.items():
            # Higher accuracy is fine, lower latency is required
            if specs["latency"] > task_req["latency"]:
                # Too slow - cannot use
                prob += model_vars[t][name] == 0
                ineligible_models.append(name)
            elif specs["accuracy"] < task_req["accuracy"]:
                # Too low accuracy - cannot use
                prob += model_vars[t][name] == 0
                ineligible_models.append(name)
            else:
                # Both requirements met
                eligible_models.append(name)

        # Battery dynamics
        if t == 0:
            # Initial battery level (start at 100%)
            prob += battery_vars[t] == 100
        else:
            # Energy consumed only if task succeeds
            energy_used_mwh = pulp.lpSum(
                [
                    available_models[name]["energy_per_task_mwh"]
                    * model_vars[t - 1][name]
                    for name in available_models.keys()
                ]
            )

            # Convert mWh to battery percentage (battery_capacity is in Wh, need to convert to mWh)
            energy_used_percent = energy_used_mwh / (battery_capacity * 1000) * 100

            # Charging: convert watts to mWh, then to percentage
            charge_added_mwh = (
                charge_vars[t - 1] * charge_rate * task_interval / 1000
            )  # W to mWh
            charge_added_percent = charge_added_mwh / (battery_capacity * 1000) * 100

            # Battery transition
            prob += (
                battery_vars[t]
                == battery_vars[t - 1] + charge_added_percent - energy_used_percent
            )

        # Prevent battery from going negative
        prob += battery_vars[t] >= 0

        # Hard constraint: if battery is 0, cannot run any models
        if t > 0:
            for name in available_models.keys():
                prob += (
                    model_vars[t][name] <= battery_vars[t - 1] / 0.01
                )  # If battery <= 0.01, model must be 0

        # Super miss: if battery is 0, we must super miss and charge
        # Simple formulation: if battery_vars[t] <= 0.1, then super_miss_vars[t] = 1
        prob += (
            super_miss_vars[t] >= 1 - battery_vars[t] / 0.1
        )  # If battery <= 0.1, super_miss >= 1

        # If super miss, we must charge to recover
        prob += charge_vars[t] >= super_miss_vars[t]

        # Battery bounds
        prob += battery_vars[t] >= 0
        prob += battery_vars[t] <= 100

        # Prevent excessive charging: if battery > 80%, discourage charging
        prob += (
            charge_vars[t] <= 1 - (battery_vars[t] - 80) / 40
        )  # Linear penalty when battery > 80%

    # Solve MILP
    print(f"🚀 Solving MILP with {num_timesteps} timesteps...")
    result = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[result]
    print(f"📈 MILP Status: {status}")

    if status != "Optimal":
        print(f"❌ MILP failed with status: {status}")
        print(
            f"🔍 Clean energy range: {min(clean_energy_series):.1f}% - {max(clean_energy_series):.1f}%"
        )
        print(
            f"🔍 Task requirements: {task_requirements[0] if task_requirements else 'None'}"
        )
        return []

    # Extract optimal schedule
    optimal_schedule = []
    action_counts = {
        "success": 0,
        "small_miss": 0,
        "large_miss": 0,
        "super_miss": 0,
        "charge": 0,
    }
    model_counts = {}

    for t in range(num_timesteps):
        selected_model = None
        should_charge = False
        status = "success"

        # Check what action was chosen
        for name, var in model_vars[t].items():
            val = pulp.value(var)
            if val is not None and abs(val - 1.0) < 1e-6:
                selected_model = name
                model_counts[name] = model_counts.get(name, 0) + 1
                break

        charge_val = pulp.value(charge_vars[t])
        if charge_val is not None and abs(charge_val - 1.0) < 1e-6:
            should_charge = True
            action_counts["charge"] += 1

        small_miss_val = pulp.value(small_miss_vars[t])
        if small_miss_val is not None and abs(small_miss_val - 1.0) < 1e-6:
            # Find closest model to criteria (may not meet requirements)
            current_task_req = task_requirements[t]

            # Calculate distance score for each model (lower is better)
            def model_distance(name, specs):
                accuracy_diff = max(
                    0, current_task_req["accuracy"] - specs["accuracy"]
                )  # How much accuracy is missing
                latency_diff = max(
                    0, specs["latency"] - current_task_req["latency"]
                )  # How much latency exceeds
                return (
                    accuracy_diff * 100 + latency_diff
                )  # Weight accuracy more heavily

            # Pick model with minimum distance to requirements
            best_model = min(
                available_models.keys(),
                key=lambda x: model_distance(x, available_models[x]),
            )
            selected_model = best_model
            status = "small_miss"
            action_counts["small_miss"] += 1

        super_miss_val = pulp.value(super_miss_vars[t])
        if super_miss_val is not None and abs(super_miss_val - 1.0) < 1e-6:
            selected_model = None
            should_charge = False
            status = "super_miss"
            action_counts["super_miss"] += 1

        if selected_model is not None and not should_charge:
            action_counts["success"] += 1

        if t < 5:  # Debug first 5 timesteps
            print(
                f"⏱️  Timestep {t}: model={selected_model}, charge={should_charge}, status={status}"
            )

        optimal_schedule.append((selected_model, should_charge, status))

    # Summary statistics
    print("📊 Schedule Summary:")
    print(f"   ✅ Success: {action_counts['success']}")
    print(f"   ⚡ Charge: {action_counts['charge']}")
    print(f"   ❌ Small Miss: {action_counts['small_miss']}")
    print(f"   💀 Super Miss: {action_counts['super_miss']}")
    print(f"   🤖 Models used: {dict(model_counts)}")

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
    """Get clean energy percentages for full day using latest available data"""
    energy_data = EnergyData()
    day = get_representative_day_for_season(season)
    location_file = get_location_filename(location)

    clean_energy_series = []
    duration_seconds = config.duration_days * 24 * 3600
    task_interval = config.task_interval_seconds

    for timestamp in range(0, int(duration_seconds), task_interval):
        dt = datetime(2024, day["month"], day["day"]) + timedelta(seconds=timestamp)

        # Use latest available clean energy data (no interpolation)
        # If we don't have exact timestamp, use the most recent available
        clean_energy_pct = energy_data.get_clean_energy_percentage(
            location_file, dt.strftime("%Y-%m-%d %H:%M:%S")
        )

        # If no data available, use the last known value
        if clean_energy_pct is None:
            clean_energy_pct = clean_energy_series[-1] if clean_energy_series else 35.0

        clean_energy_series.append(clean_energy_pct)

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
    t: int, optimal_schedule: List[Tuple[str, bool, str]], config: SimulationConfig
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
            "energy_per_task_mwh": profile[
                "energy_per_inference_mwh"
            ],  # Energy per task in mWh
        }

    battery_level = 100.0  # Start at 100%
    task_interval = config.task_interval_seconds
    battery_capacity = config.battery_capacity_wh
    charge_rate = config.charge_rate_watts

    for i in range(t):
        model_name, should_charge, status = optimal_schedule[i]

        # Only consume energy for both successful tasks and small misses
        if status in ["success", "small_miss"] and model_name is not None:
            energy_used_mwh = available_models[model_name]["energy_per_task_mwh"]
        else:
            energy_used_mwh = 0  # No energy consumption for super misses or None models

        # Convert mWh to battery percentage (battery_capacity is in Wh, need to convert to mWh)
        energy_used_percent = energy_used_mwh / (battery_capacity * 1000) * 100
        battery_level -= energy_used_percent

        # Charge if needed
        if should_charge:
            # Charging: convert watts to mWh, then to percentage
            charge_added_mwh = charge_rate * task_interval / 1000  # W to mWh
            charge_added_percent = charge_added_mwh / (battery_capacity * 1000) * 100
            battery_level += charge_added_percent

        # Ensure bounds
        battery_level = max(0, min(100, battery_level))

        # If battery would go negative from this action, set to 0 and indicate no model can run
        if battery_level == 0 and i < t:
            # This means we ran out of battery before current timestep
            return 0.0

    return battery_level


def generate_random_training_samples(
    work_item: Tuple[str, str, int], config: SimulationConfig
) -> List[Dict]:
    """Generate random training samples using short-horizon MILP problems"""
    location, season, num_samples = work_item

    logger.info(f"Generating {num_samples} random samples for {location} {season}")

    # Extract full day's clean energy series for sampling
    full_day_clean_energy = get_clean_energy_series(location, season, config)
    full_day_task_requirements = generate_task_requirements(config)

    # Generate random short MILP problems
    import random

    random.seed(hash(f"{location}_{season}"))  # Location-season specific seed

    training_examples = []
    horizon_length = 100  # 100 timesteps = ~8.3 minutes per MILP problem

    for i in range(num_samples):
        # Random start time in the day
        start_timestep = random.randint(
            0, len(full_day_clean_energy) - horizon_length - 1
        )

        # Extract short horizon
        clean_energy_slice = full_day_clean_energy[
            start_timestep : start_timestep + horizon_length
        ]
        task_requirements_slice = full_day_task_requirements[
            start_timestep : start_timestep + horizon_length
        ]

        # Solve short MILP (much faster than full day)
        optimal_schedule = solve_full_horizon_milp(
            clean_energy_slice, task_requirements_slice, config
        )

        # Take the first decision from the optimal schedule
        if optimal_schedule:
            t = 0  # First timestep decision
            model, charge, status = optimal_schedule[t]

            # Use the actual battery level from MILP solution (starts at 50%)
            battery_level = 50.0  # MILP always starts at 50%

            example = {
                "battery_level": battery_level,
                "clean_energy_percentage": clean_energy_slice[t],
                "accuracy_requirement": task_requirements_slice[t]["accuracy"],
                "latency_requirement": task_requirements_slice[t]["latency"],
                "optimal_model": model,
                "should_charge": charge,
                "task_status": status,
            }
            training_examples.append(example)

    logger.info(
        f"Generated {len(training_examples)} training examples for {location} {season}"
    )
    return training_examples


def generate_day_training_data(
    work_item: Tuple[str, str, str], config: SimulationConfig
) -> List[Dict]:
    """Generate training data for one day using full-horizon MILP"""
    location, season, day_str = work_item

    print(f"🌍 Generating training data for {location} {season} {day_str}")
    print(
        f"📅 Processing {config.duration_days} day(s) with {config.task_interval_seconds}s intervals"
    )

    # Extract full day's clean energy series
    print("⚡ Loading clean energy data...")
    clean_energy_series = get_clean_energy_series(location, season, config)
    print(
        f"📈 Clean energy range: {min(clean_energy_series):.1f}% - {max(clean_energy_series):.1f}%"
    )

    # Generate task requirements for day
    print("🎯 Generating task requirements...")
    task_requirements = generate_task_requirements(config)
    print(
        f"📋 Task requirements: {task_requirements[0] if task_requirements else 'None'}"
    )

    # Solve full-horizon MILP
    print("🧠 Starting MILP optimization...")
    optimal_schedule = solve_full_horizon_milp(
        clean_energy_series, task_requirements, config
    )

    if not optimal_schedule:
        print(f"❌ No optimal schedule generated for {location} {season}")
        return []

    # Extract training examples from optimal schedule
    print("📝 Extracting training examples...")
    training_examples = []
    for t, (model, charge, status) in enumerate(optimal_schedule):
        # Calculate battery level BEFORE this timestep's action
        battery_before_action = calculate_battery_at_timestep(
            t, optimal_schedule, config
        )

        # If battery is 0, override status to task_miss or super_miss
        actual_status = status
        if battery_before_action <= 0.01 and status == "success" and model is not None:
            actual_status = "large_miss"  # Cannot run models with no battery (Large Miss per README)

        example = {
            "battery_level": battery_before_action,
            "clean_energy_percentage": clean_energy_series[t],
            "accuracy_requirement": task_requirements[t]["accuracy"],
            "latency_requirement": task_requirements[t]["latency"],
            "optimal_model": model,
            "should_charge": charge,
            "task_status": actual_status,
        }
        training_examples.append(example)

    print(
        f"✅ Generated {len(training_examples)} training examples for {location} {season}"
    )
    return training_examples


def generate_full_horizon_training_data() -> List[Dict]:
    """Generate training data using full-horizon optimization for 4 days (1 per season)."""
    print("=" * 80)
    print("🚀 STARTING FULL-HORIZON TRAINING DATA GENERATION")
    print("=" * 80)

    # Load configuration
    config_loader = ConfigLoader()
    config = config_loader.get_simulation_config()
    workers_config = config_loader.get_workers_config()

    max_workers = workers_config.max_workers
    print(f"🔧 Using {max_workers} workers for parallel processing")

    # Use just one location (CA) and 4 seasons = 4 full days
    location = "CA"
    work_items = []
    for season in config.seasons:  # ["winter", "spring", "summer", "fall"]
        day = get_representative_day_for_season(season)
        day_str = f"{day['month']:02d}-{day['day']:02d}"
        work_items.append((location, season, day_str))

    print(f"📅 Processing {len(work_items)} full days for {location}")
    print(f"🌟 Seasons: {config.seasons}")

    # Parallel processing using config workers
    print("⚡ Starting parallel processing...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(generate_day_training_data, item, config)
            for item in work_items
        ]

        # Collect results
        all_training_data = []
        completed = 0
        for future in as_completed(futures):
            try:
                day_data = future.result()
                all_training_data.extend(day_data)
                completed += 1
                print(
                    f"✅ Completed {completed}/{len(work_items)} days ({len(day_data)} examples)"
                )
            except Exception as e:
                print(f"❌ Error processing training data: {e}")
                import traceback

                traceback.print_exc()

    print(f"🎉 Generated {len(all_training_data)} total training examples")
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
        output_file = results_dir / "training_data.json"
        print(f"DEBUG: Saving {len(training_data)} examples to {output_file}")
        with open(output_file, "w") as f:
            json.dump(training_data, f, indent=2)
        print("DEBUG: File saved successfully")

        print(f"✓ Saved {len(training_data)} training examples to {output_file}")
        logger.info(f"✓ Saved {len(training_data)} training examples to {output_file}")

    except Exception as e:
        print(f"✗ Error generating training data: {e}")
        logger.error(f"✗ Error generating training data: {e}")
        logger.exception("Full traceback for training data generation error")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
