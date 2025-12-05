#!/usr/bin/env python3
"""
Quick test to verify pipeline fix
"""

import sys
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))


def test_simulation_runner_fix():
    """Test that the simulation runner fix works."""
    try:
        from simulation_runner import BasicSimulationRunner

        print("Testing simulation runner fix...")

        # Create runner - this should work now
        runner = BasicSimulationRunner(config_path="config.jsonc", max_workers=1)
        print("✓ BasicSimulationRunner initialized without error")

        # Test that we can get the number of simulations
        locations = runner.config_loader.get_locations()
        seasons = runner.config_loader.get_seasons()
        controllers = runner.config_loader.get_controllers()

        simulations = runner._generate_simulation_list(
            locations=locations, seasons=seasons, controllers=controllers, weeks=[1]
        )

        print(f"✓ Can generate {len(simulations)} simulations")
        print("✓ Fix successful - no more simulation_configs attribute error")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_simulation_runner_fix()
    print(f"\n{'✅ Test passed!' if success else '❌ Test failed!'}")
    sys.exit(0 if success else 1)
