#!/usr/bin/env python3
"""
Test to verify battery charging/discharging logic works correctly
"""

import sys
import logging
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.controller import CustomController
from src.simulation_engine import SimulationEngine
from src.config_loader import ConfigLoader
from src.energy_data import EnergyData

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def test_battery_simulation():
    """Test that battery properly charges and discharges in simulation."""
    print("=== Testing Battery Simulation ===")

    # Load config
    config_loader = ConfigLoader("config.jsonc")
    config = config_loader.get_simulation_config()

    # Load power profiles
    import json

    with open("results/power_profiles.json", "r") as f:
        power_profiles = json.load(f)

    # Create controller
    controller = CustomController()

    # Create energy data
    energy_data = EnergyData()

    # Create simulation engine
    engine = SimulationEngine(
        config=config,
        controller=controller,
        location="CA",
        season="summer",
        week=1,
        power_profiles=power_profiles,
        energy_data=energy_data,
    )

    print(f"Initial battery: {engine.battery.get_percentage():.1f}%")

    # Run simulation for a few steps to verify battery behavior
    for i in range(20):
        # Generate task
        task = engine.task_generator.generate_task(
            timestamp=i * 5,
            accuracy_req=config.user_accuracy_requirement,
            latency_req=config.user_latency_requirement,
        )

        if task:
            # Get controller decision
            battery_level_before = engine.battery.get_percentage()
            clean_energy_pct = engine._get_clean_energy_percentage(task.timestamp)
            available_models = engine._get_available_models()

            choice = controller.select_model(
                battery_level=battery_level_before,
                clean_energy_percentage=clean_energy_pct,
                user_accuracy_requirement=task.accuracy_requirement,
                user_latency_requirement=task.latency_requirement,
                available_models=available_models,
            )

            # Execute task (discharge)
            model_specs = available_models[choice.model_name]
            power_mw = model_specs["power_cost"]
            duration_seconds = model_specs["latency"]

            engine.battery.discharge(
                power_mw=power_mw,
                duration_seconds=duration_seconds,
                clean_energy_percentage=clean_energy_pct,
            )

            battery_level_after_discharge = engine.battery.get_percentage()

            # Handle charging
            if choice.should_charge:
                engine.battery.charge(config.task_interval_seconds)
                battery_level_final = engine.battery.get_percentage()
                print(
                    f"Step {i + 1}: Battery {battery_level_before:.1f}% -> {battery_level_after_discharge:.1f}% (discharge) -> {battery_level_final:.1f}% (after charge), Model: {choice.model_name}, Charged: {choice.should_charge}"
                )
            else:
                battery_level_final = battery_level_after_discharge
                print(
                    f"Step {i + 1}: Battery {battery_level_before:.1f}% -> {battery_level_final:.1f}% (discharge only), Model: {choice.model_name}, Charged: {choice.should_charge}"
                )

            # Stop if battery gets too low
            if battery_level_final < 10.0:
                print("Battery too low, stopping test")
                break
        else:
            print(f"Step {i + 1}: No task generated")

    print(f"Final battery: {engine.battery.get_percentage():.1f}%")
    print("✓ Battery simulation test completed")


if __name__ == "__main__":
    test_battery_simulation()
