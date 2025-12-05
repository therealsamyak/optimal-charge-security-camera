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


def load_energy_data() -> Dict[str, pd.DataFrame]:
    """Load energy data from all 4 locations."""
    energy_dir = Path("energy-data")
    energy_data = {}
    
    # Find all CSV files in energy-data directory
    csv_files = list(energy_dir.glob("*.csv"))
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, dtype=str, low_memory=False)
            df["Datetime (UTC)"] = pd.to_datetime(df["Datetime (UTC)"])
            # Use filename without extension as location key
            location = csv_file.stem
            energy_data[location] = df
            print(f"Loaded {len(df)} records for {location}")
        except Exception as e:
            print(f"Failed to load {csv_file}: {e}")
            
    return energy_data


def get_clean_energy_percentage(energy_data: Dict[str, pd.DataFrame], 
                              location: str, timestamp: datetime) -> float:
    """Get clean energy percentage for specific location and timestamp."""
    if location not in energy_data:
        return 50.0  # Default fallback
    
    df = energy_data[location]
    
    # Find closest data point
    past_data = df[df["Datetime (UTC)"] <= timestamp]
    if past_data.empty:
        return 50.0
    
    latest_row = past_data.iloc[-1]
    cfe_col = "Carbon-free energy percentage (CFE%)"
    
    if cfe_col in latest_row:
        try:
            return float(latest_row[cfe_col])
        except (ValueError, TypeError):
            pass
    
    # Handle missing data by averaging between before/after
    future_data = df[df["Datetime (UTC)"] > timestamp]
    if not future_data.empty:
        next_row = future_data.iloc[0]
        if cfe_col in next_row:
            try:
                before_val = float(latest_row.get(cfe_col, 50))
                after_val = float(next_row[cfe_col])
                return (before_val + after_val) / 2.0
            except (ValueError, TypeError):
                pass
    
    return 50.0  # Default fallback


def load_power_profiles() -> Dict[str, Dict[str, float]]:
    """Load power profiles using PowerProfiler."""
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    
    from src.power_profiler import PowerProfiler
    profiler = PowerProfiler()
    profiler.load_profiles()  # Load profiles from file
    
    # Transform profiles to match expected field names
    raw_profiles = profiler.get_all_models_data()
    transformed_profiles = {}
    
    for name, profile in raw_profiles.items():
        transformed_profiles[name] = {
            "accuracy": profile["accuracy"],
            "latency": profile["avg_inference_time_seconds"] * 1000,  # Convert to ms
            "power_cost": profile["model_power_mw"],  # Power in mW
        }
    
    return transformed_profiles


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
    prob = pulp.LpProblem("Training_Scenario", pulp.LpMaximize)

    model_vars = {
        name: pulp.LpVariable(f"use_{name}", cat="Binary")
        for name in available_models.keys()
    }
    charge_var = pulp.LpVariable("charge", cat="Binary")

    # Objective: Maximize clean energy usage
    # Focus on charging when clean energy is high
    prob += clean_energy_percentage * charge_var

    # Constraint: Select exactly one model
    prob += pulp.lpSum(model_vars.values()) == 1

    # Filter models based on requirements
    for name, specs in available_models.items():
        if specs["accuracy"] < accuracy_requirement:
            prob += model_vars[name] == 0
        if specs["latency"] > latency_requirement:
            prob += model_vars[name] == 0

    # Battery capacity constraint
    prob += battery_level + charge_var * 15 <= 100

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    selected_model = list(available_models.keys())[0]
    for name, var in model_vars.items():
        if pulp.value(var) == 1:
            selected_model = name
            break

    should_charge = pulp.value(charge_var) == 1

    return selected_model, should_charge


def solve_scenario_wrapper(scenario_data: Tuple[int, int, float, int, str, datetime]) -> Optional[Dict]:
    """Wrapper function for parallel execution of scenario solving."""
    battery, clean_energy, acc_req, lat_req, location, timestamp = scenario_data
    
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
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    locations = list(energy_data.keys())
    scenarios = []
    
    # Generate 100,000 scenarios with real energy data
    print("Generating 100,000 training scenarios...")
    for i in range(100000):
        if i % 20000 == 0:
            print(f"Scenario generation progress: {i}/100000 ({i/1000:.0f}%)")
        # Random location for maximal coverage
        location = random.choice(locations)
        
        # Random timestamp throughout the year
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 12, 31)
        random_timestamp = start_date + timedelta(
            seconds=random.randint(0, int((end_date - start_date).total_seconds()))
        )
        
        # Get real clean energy percentage
        clean_energy = get_clean_energy_percentage(energy_data, location, random_timestamp)
        
        # Debug progress every 1000 scenarios
        if i % 1000 == 0 and i > 0:
            print(f"Scenario generation progress: {i}/100000 ({i/1000:.0f}%)")
        
        # Random other parameters
        battery = np.random.uniform(1, 100)
        acc_req = np.random.uniform(0.3, 1.0)
        lat_req = np.random.choice([1, 2, 3, 5, 8, 10, 15, 20, 25, 30])
        
        scenarios.append((battery, clean_energy, acc_req, lat_req, location, random_timestamp))

    return scenarios


def main():
    """Generate training data and save to JSON."""
    print("Loading energy data...")
    try:
        energy_data = load_energy_data()
        print(f"✓ Loaded energy data for {len(energy_data)} regions")
    except Exception as e:
        print(f"✗ Error loading energy data: {e}")
        return
    
    print("Loading power profiles...")
    try:
        models = load_power_profiles()
        print(f"✓ Loaded {len(models)} model profiles")
        if not models:
            print("✗ No model profiles found - power_profiles.json may be empty")
            return
    except Exception as e:
        print(f"✗ Error loading power profiles: {e}")
        return

    print("Generating training scenarios with real energy data...")
    try:
        scenarios = generate_training_scenarios(energy_data)
        print(f"✓ Generated {len(scenarios)} training scenarios")
    except Exception as e:
        print(f"✗ Error generating scenarios: {e}")
        return

    print(f"Solving MIPS for {len(scenarios)} scenarios in parallel...")
    training_data = []
    # Use ProcessPoolExecutor for parallel execution
    max_workers = 20
    print(f"Starting parallel processing with {max_workers} workers...")
    print(f"Submitting {len(scenarios)} scenarios to worker pool...")
    
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all scenarios for processing
            future_to_scenario = {
                executor.submit(solve_scenario_wrapper, scenario): i
                for i, scenario in enumerate(scenarios)
            }
            print(f"✓ All {len(scenarios)} scenarios submitted to workers")

            completed_count = 0
            error_count = 0
            # Collect results as they complete
            print("Waiting for scenarios to complete...")
            for future in concurrent.futures.as_completed(future_to_scenario):
                scenario_index = future_to_scenario[future]
                completed_count += 1
                
                # Progress updates more frequently for better visibility
                if completed_count % 100 == 0:
                    progress_pct = (completed_count / len(scenarios)) * 100
                    print(f"Progress: {completed_count}/{len(scenarios)} ({progress_pct:.1f}%) - Errors: {error_count}")
                elif completed_count % 25 == 0:
                    progress_pct = (completed_count / len(scenarios)) * 100
                    print(f"Progress: {completed_count}/{len(scenarios)} ({progress_pct:.1f}%)")
                
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
                    continue
                    
        print(f"✓ Parallel processing completed - {len(training_data)} successful, {error_count} failed")
    except Exception as e:
        print(f"✗ Error in parallel processing: {e}")
        return

    print(f"Solving MIPS for {len(scenarios)} scenarios in parallel...")
    training_data = []

    # Use ProcessPoolExecutor for parallel execution
    max_workers = 20
    print(f"Starting parallel processing with {max_workers} workers...")
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all scenarios for processing
            future_to_scenario = {
                executor.submit(solve_scenario_wrapper, scenario): i
                for i, scenario in enumerate(scenarios)
            }

            completed_count = 0
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_scenario):
                scenario_index = future_to_scenario[future]
                completed_count += 1
                
                if completed_count % 1000 == 0:
                    print(f"Progress: {completed_count}/{len(scenarios)}")
                
                try:
                    result = future.result()
                    if result:
                        training_data.append(result)
                except Exception as e:
                    print(f"Error processing scenario {scenario_index}: {e}")
                    continue
    except Exception as e:
        print(f"✗ Error in parallel processing: {e}")
        return

    print(f"✓ Generated {len(training_data)} training samples")

    # Save to JSON
    try:
        with open("results/training_data.json", "w") as f:
            json.dump(training_data, f, indent=2)
        print("✓ Training data saved to results/training_data.json")
    except Exception as e:
        print(f"✗ Error saving training data: {e}")
        return


if __name__ == "__main__":
    main()