#!/usr/bin/env python3
"""
Quick test to verify battery logic works with fixed CustomController
"""

import sys
import logging
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.simulation_engine import SimulationEngine
from src.config_loader import ConfigLoader
from src.energy_data import EnergyData

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def test_single_simulation():
    """Test a single simulation to verify battery behavior."""
    print("=== Testing Single Simulation ===")

    # Load config
    config_loader = ConfigLoader("config.jsonc")
    config = config_loader.get_simulation_config()

    # Load power profiles
    import json

    with open("results/power_profiles.json", "r") as f:
        power_profiles = json.load(f)

    # Create custom controller
    from src.controller import CustomController

    controller = CustomController()

    # Create energy data
    energy_data = EnergyData()

    # Create simulation engine with short duration for testing
    config.duration_days = 1  # 1 day for testing
    config.time_acceleration = 1000  # Very fast

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

    # Run simulation
    metrics = engine.run()

    print(f"Final battery: {engine.battery.get_percentage():.1f}%")
    print(f"Total tasks: {metrics['total_tasks']}")
    print(f"Completed tasks: {metrics['completed_tasks']}")
    print(f"Task completion rate: {metrics.get('task_completion_rate', 0):.1f}%")

    # Check battery levels during simulation
    if metrics["battery_levels"]:
        levels = [entry["level"] for entry in metrics["battery_levels"]]
        min_level = min(levels)
        max_level = max(levels)
        print(f"Battery range: {min_level:.1f}% to {max_level:.1f}%")

        # Check if battery ever charged when full
        full_charges = 0
        for i in range(1, len(levels)):
            if levels[i - 1] >= 99.5 and levels[i] > levels[i - 1]:
                full_charges += 1
        print(f"Times battery charged while nearly full: {full_charges}")

    print("✓ Single simulation test completed")


if __name__ == "__main__":
    test_single_simulation()
