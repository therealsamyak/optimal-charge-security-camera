#!/usr/bin/env python3
"""
Test for BasicSimulationRunner initialization issue
"""

import sys
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))


def test_basic_runner_initialization():
    """Test that BasicSimulationRunner initializes correctly."""
    try:
        from src.simulation_runner_base import SimulationRunnerBase
        from src.config_loader import ConfigLoader

        print("Testing BasicSimulationRunner initialization...")

        # Test config loading
        config_loader = ConfigLoader("config.jsonc")
        print("✓ ConfigLoader initialized")

        # Test base class initialization
        runner = SimulationRunnerBase("config.jsonc", max_workers=1)
        print("✓ SimulationRunnerBase initialized")

        # Check what attributes are available
        print(
            f"Available attributes: {[attr for attr in dir(runner) if not attr.startswith('_')]}"
        )

        # Test simulation generation
        locations = config_loader.get_locations()
        seasons = config_loader.get_seasons()
        controllers = config_loader.get_controllers()

        print(f"Locations: {locations}")
        print(f"Seasons: {seasons}")
        print(f"Controllers: {controllers}")

        simulations = runner._generate_simulation_list(
            locations=locations, seasons=seasons, controllers=controllers, weeks=[1]
        )

        print(f"✓ Generated {len(simulations)} simulations")
        print(f"First simulation: {simulations[0] if simulations else 'None'}")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_basic_runner_initialization()
    sys.exit(0 if success else 1)
