#!/usr/bin/env python3
"""
MIPS solver to generate training data for CustomController.
Generates optimal decisions for diverse scenarios and caches to JSON.
"""

import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pulp
import concurrent.futures


def load_power_profiles() -> Dict[str, Dict[str, float]]:
    """Load power profiles from results and model data from CSV."""
    with open("results/power_profiles.json", "r") as f:
        profiles = json.load(f)

    # Load real model data
    model_data = {}
    with open("model-data/model-data.csv", "r") as f:
        lines = f.readlines()
        for line in lines[1:]:  # Skip header
            parts = line.strip().split(",")
            model = parts[0].strip('"')
            version = parts[1].strip('"')
            latency = float(parts[2].strip('"'))
            accuracy = float(parts[3].strip('"'))
            model_data[f"{model}_{version}"] = {
                "accuracy": accuracy,
                "latency": latency,
            }

    models = {}
    for model_name, data in profiles.items():
        # Use real accuracy and latency from model-data.csv
        real_data = model_data.get(model_name, {})
        models[model_name] = {
            "accuracy": real_data.get("accuracy", 85.0),  # Fallback to 85% if not found
            "latency": real_data.get(
                "latency", data["avg_inference_time_seconds"] * 1000
            ),  # Use real latency, fallback to power profile
            "power_cost": data[
                "model_power_mw"
            ],  # Keep power data from power profiling
        }

    return models


def solve_mips_scenario(
    battery_level: float,
    clean_energy_percentage: float,
    accuracy_requirement: float,
    latency_requirement: float,
    available_models: Dict[str, Dict[str, float]],
) -> Tuple[str, bool]:
    """
    Solve MIPS for a single scenario to get optimal model and charging decision.
    """
    prob = pulp.LpProblem("Training_Scenario", pulp.LpMaximize)

    model_vars = {
        name: pulp.LpVariable(f"use_{name}", cat="Binary")
        for name in available_models.keys()
    }
    charge_var = pulp.LpVariable("charge", cat="Binary")

    # Normalize all objectives to 0-1 range for equal weighting
    max_accuracy = max(specs["accuracy"] for specs in available_models.values())
    min_accuracy = min(specs["accuracy"] for specs in available_models.values())
    max_latency = max(specs["latency"] for specs in available_models.values())
    min_latency = min(specs["latency"] for specs in available_models.values())
    
    # Smart charging logic in objective function
    # Add bonus for charging when battery is low
    battery_urgency_bonus = max(0, (50 - battery_level) / 50.0)  # 0-1 range
    
    # Add bonus for charging during high clean energy
    clean_energy_bonus = clean_energy_percentage / 100.0  # 0-1 range
    
    # Penalty for charging when battery is already high
    battery_waste_penalty = max(0, (battery_level - 80) / 20.0) if battery_level > 80 else 0  # 0-1 range
    
    prob += (
        # Normalized accuracy (0-1 range, higher is better)
        pulp.lpSum(
            [
                ((available_models[name]["accuracy"] - min_accuracy) / (max_accuracy - min_accuracy)) * model_vars[name]
                for name in available_models.keys()
            ]
        )
        # Normalized latency (0-1 range, lower is better, so subtract)
        - pulp.lpSum(
            [
                ((available_models[name]["latency"] - min_latency) / (max_latency - min_latency)) * model_vars[name]
                for name in available_models.keys()
            ]
        )
        # Smart charging decision (combines urgency, opportunity, and waste avoidance)
        + (battery_urgency_bonus * 0.4 + clean_energy_bonus * 0.4 - battery_waste_penalty * 0.2) * charge_var
    )

    prob += pulp.lpSum(model_vars.values()) == 1

    # Filter models based on requirements with edge cases
    min_accuracy_model = min(available_models.keys(), key=lambda x: available_models[x]["accuracy"])
    max_accuracy_model = max(available_models.keys(), key=lambda x: available_models[x]["accuracy"])
    min_accuracy = available_models[min_accuracy_model]["accuracy"]
    max_accuracy = available_models[max_accuracy_model]["accuracy"]
    
    # If requirement is higher than highest model, force use of highest model
    if accuracy_requirement > max_accuracy:
        for name in available_models.keys():
            if name != max_accuracy_model:
                prob += model_vars[name] == 0
    # If requirement is lower than lowest model, allow any model (no filtering)
    elif accuracy_requirement < min_accuracy:
        pass  # No accuracy filtering, all models eligible
    else:
        # Normal filtering based on accuracy requirement
        for name, specs in available_models.items():
            if specs["accuracy"] < accuracy_requirement:
                prob += model_vars[name] == 0
    
    # Apply latency filtering to all cases
    for name, specs in available_models.items():
        if specs["latency"] > latency_requirement:
            prob += model_vars[name] == 0

    # Battery capacity constraint
    prob += battery_level + charge_var * 15 <= 100
    
    # Smart charging logic in objective function
    # Add bonus for charging when battery is low
    battery_urgency_bonus = max(0, (50 - battery_level) / 50.0)  # 0-1 range
    
    # Add bonus for charging during high clean energy
    clean_energy_bonus = clean_energy_percentage / 100.0  # 0-1 range
    
    # Penalty for charging when battery is already high
    battery_waste_penalty = max(0, (battery_level - 80) / 20.0) if battery_level > 80 else 0  # 0-1 range

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    selected_model = list(available_models.keys())[0]
    for name, var in model_vars.items():
        if pulp.value(var) == 1:
            selected_model = name
            break

    should_charge = pulp.value(charge_var) == 1

    return selected_model, should_charge


def solve_scenario_wrapper(scenario_data: Tuple[int, int, float, int]) -> Optional[Dict]:
    """Wrapper function for parallel execution of scenario solving."""
    battery, clean_energy, acc_req, lat_req = scenario_data
    
    # Load models inside the worker process to avoid serialization issues
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
            "optimal_model": selected_model,
            "should_charge": bool(should_charge),
        }
    except Exception as e:
        print(f"Error solving scenario: {e}")
        return None


def generate_training_scenarios(
    seed: Optional[int] = None,
) -> List[Tuple[int, int, float, int]]:
    """Generate purely random training scenarios with optional seed."""
    if seed is not None:
        np.random.seed(seed)

    # Generate 100,000 purely random scenarios
    scenarios = []
    for _ in range(100000):
        battery = np.random.uniform(1, 100)
        clean_energy = np.random.uniform(0, 100)
        acc_req = np.random.uniform(0.3, 1.0)
        lat_req = np.random.choice([1, 2, 3, 5, 8, 10, 15, 20, 25, 30])
        scenarios.append((battery, clean_energy, acc_req, lat_req))

    return scenarios


def main():
    """Generate training data and save to JSON."""
    print("Loading power profiles...")
    models = load_power_profiles()

    print("Generating training scenarios...")
    scenarios = generate_training_scenarios()

    print(f"Solving MIPS for {len(scenarios)} scenarios in parallel...")
    training_data = []

    # Use ProcessPoolExecutor for parallel execution
    max_workers = 20  # Adjust based on your CPU cores
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

    print(f"Generated {len(training_data)} training samples")

    # Save to JSON
    with open("results/training_data.json", "w") as f:
        json.dump(training_data, f, indent=2)

    print("Training data saved to results/training_data.json")


if __name__ == "__main__":
    main()
