#!/usr/bin/env python3
"""
Test BasicSimulationRunner functionality without file I/O
"""

import sys
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))


def test_basic_simulation_runner():
    """Test BasicSimulationRunner can run without errors."""
    try:
        from simulation_runner import BasicSimulationRunner

        print("Testing BasicSimulationRunner...")

        # Create runner
        runner = BasicSimulationRunner(
            config_path="config.jsonc",
            max_workers=1,
        )
        print("✓ BasicSimulationRunner initialized")

        # Test simulation generation (don't actually run)
        locations = runner.config_loader.get_locations()
        seasons = runner.config_loader.get_seasons()
        controllers = runner.config_loader.get_controllers()

        simulations = runner._generate_simulation_list(
            locations=locations, seasons=seasons, controllers=controllers, weeks=[1]
        )

        print(f"✓ Generated {len(simulations)} simulations")

        # Test that we can create a controller (without running full simulation)
        controller = runner._create_controller("naive_weak")
        print(f"✓ Created controller: {type(controller).__name__}")

        # Test summary stats with empty results
        stats = runner.get_summary_stats()
        print(f"✓ Summary stats (empty): {stats}")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_basic_simulation_runner()
    sys.exit(0 if success else 1)
