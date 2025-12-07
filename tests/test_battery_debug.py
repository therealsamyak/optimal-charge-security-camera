#!/usr/bin/env python3
"""
Test script to debug battery charging/discharging issue
"""

import sys
import logging
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.battery import Battery
from src.controller import NaiveWeakController, NaiveStrongController, CustomController
from src.simulation_engine import SimulationEngine
from src.config_loader import ConfigLoader
from src.energy_data import EnergyData

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def test_battery_logic():
    """Test battery charging/discharging logic directly."""
    print("=== Testing Battery Logic ===")

    # Create battery
    battery = Battery(capacity_wh=5.0, charge_rate_watts=100.0)
    print(f"Initial battery level: {battery.get_percentage():.1f}%")

    # Test discharge
    print("\nTesting discharge...")
    success = battery.discharge(
        power_mw=1000.0, duration_seconds=1.0
    )  # 1W for 1 second
    print(f"Discharge successful: {success}")
    print(f"Battery level after discharge: {battery.get_percentage():.1f}%")

    # Test charge
    print("\nTesting charge...")
    energy_added = battery.charge(duration_seconds=1.0)
    print(f"Energy added: {energy_added:.6f} Wh")
    print(f"Battery level after charge: {battery.get_percentage():.1f}%")

    # Test charge when full
    print("\nTesting charge when full...")
    energy_added = battery.charge(duration_seconds=10.0)
    print(f"Energy added when full: {energy_added:.6f} Wh")
    print(f"Battery level: {battery.get_percentage():.1f}%")


def test_controller_logic():
    """Test controller charging decisions."""
    print("\n=== Testing Controller Logic ===")

    # Create controllers
    weak_controller = NaiveWeakController()
    strong_controller = NaiveStrongController()

    try:
        custom_controller = CustomController()
    except Exception as e:
        print(f"Could not load CustomController: {e}")
        custom_controller = None

    # Test at different battery levels
    battery_levels = [100.0, 90.0, 50.0, 20.0, 10.0]
    clean_energy_pct = 50.0
    accuracy_req = 45.0
    latency_req = 8.0
    available_models = {
        "YOLOv10_N": {"accuracy": 50.0, "latency": 5.0, "power_cost": 500.0},
        "YOLOv10_X": {"accuracy": 80.0, "latency": 15.0, "power_cost": 2000.0},
    }

    for level in battery_levels:
        weak_choice = weak_controller.select_model(
            level, clean_energy_pct, accuracy_req, latency_req, available_models
        )
        strong_choice = strong_controller.select_model(
            level, clean_energy_pct, accuracy_req, latency_req, available_models
        )

        print(f"\nBattery {level:.1f}%:")
        print(
            f"  Weak controller - Model: {weak_choice.model_name}, Charge: {weak_choice.should_charge}"
        )
        print(
            f"  Strong controller - Model: {strong_choice.model_name}, Charge: {strong_choice.should_charge}"
        )

        if custom_controller:
            try:
                # Debug the charge score calculation
                features = [
                    level / 100.0,
                    clean_energy_pct / 100.0,
                    accuracy_req,
                    latency_req / 3000.0,
                ]
                charge_score = sum(
                    f * w for f, w in zip(features, custom_controller.charge_weights)
                )
                print(
                    f"  Custom controller debug - features: {features}, charge_score: {charge_score:.6f}, threshold: {custom_controller.charge_threshold}"
                )

                custom_choice = custom_controller.select_model(
                    level, clean_energy_pct, accuracy_req, latency_req, available_models
                )
                print(
                    f"  Custom controller - Model: {custom_choice.model_name}, Charge: {custom_choice.should_charge}"
                )
            except Exception as e:
                print(f"  Custom controller - Error: {e}")


def test_simple_simulation():
    """Test a simple simulation to see what's happening."""
    print("\n=== Testing Simple Simulation ===")

    # Load config
    config_loader = ConfigLoader("config.jsonc")
    config = config_loader.get_simulation_config()

    # Create simple power profiles
    power_profiles = {
        "YOLOv10_N": {
            "accuracy": 50.0,
            "avg_inference_time_seconds": 0.1,
            "model_power_mw": 500.0,
        }
    }

    # Create controller
    controller = NaiveWeakController()

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

    # Run a few steps manually
    for i in range(5):
        print(f"\n--- Step {i + 1} ---")

        # Generate task
        task = engine.task_generator.generate_task(
            timestamp=i * 5,
            accuracy_req=config.user_accuracy_requirement,
            latency_req=config.user_latency_requirement,
        )

        if task:
            print(
                f"Task generated: accuracy_req={task.accuracy_requirement}, latency_req={task.latency_requirement}"
            )

            # Get controller decision
            battery_level = engine.battery.get_percentage()
            clean_energy_pct = engine._get_clean_energy_percentage(task.timestamp)
            available_models = engine._get_available_models()

            choice = controller.select_model(
                battery_level=battery_level,
                clean_energy_percentage=clean_energy_pct,
                user_accuracy_requirement=task.accuracy_requirement,
                user_latency_requirement=task.latency_requirement,
                available_models=available_models,
            )

            print(
                f"Controller choice: model={choice.model_name}, should_charge={choice.should_charge}"
            )
            print(f"Battery before: {battery_level:.1f}%")

            # Execute task (discharge)
            model_specs = available_models[choice.model_name]
            power_mw = model_specs["power_cost"]
            duration_seconds = model_specs["latency"]

            success = engine.battery.discharge(
                power_mw=power_mw,
                duration_seconds=duration_seconds,
                clean_energy_percentage=clean_energy_pct,
            )

            print(f"Discharge successful: {success}")
            print(f"Battery after discharge: {engine.battery.get_percentage():.1f}%")

            # Handle charging
            if choice.should_charge:
                energy_added = engine.battery.charge(config.task_interval_seconds)
                print(f"Charging: energy_added={energy_added:.6f} Wh")
                print(f"Battery after charge: {engine.battery.get_percentage():.1f}%")
        else:
            print("No task generated")


if __name__ == "__main__":
    test_battery_logic()
    test_controller_logic()
    test_simple_simulation()
