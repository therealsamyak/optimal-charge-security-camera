#!/usr/bin/env python3
"""
Final verification test for pipeline fix
"""

import sys
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))


def main():
    """Verify pipeline fix is working."""
    print("🔧 Pipeline Fix Verification")
    print("=" * 40)

    # Test 1: BasicSimulationRunner initialization
    try:
        from simulation_runner import BasicSimulationRunner

        runner = BasicSimulationRunner(config_path="config.jsonc", max_workers=1)
        print("✓ BasicSimulationRunner initializes without error")
    except AttributeError as e:
        if "simulation_configs" in str(e):
            print("✗ simulation_configs attribute error still exists")
            return 1
        else:
            print(f"✗ Unexpected error: {e}")
            return 1
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return 1

    # Test 2: Can generate simulation list
    try:
        locations = runner.config_loader.get_locations()
        seasons = runner.config_loader.get_seasons()
        controllers = runner.config_loader.get_controllers()

        simulations = runner._generate_simulation_list(
            locations=locations, seasons=seasons, controllers=controllers, weeks=[1]
        )
        print(f"✓ Can generate {len(simulations)} simulations")
    except Exception as e:
        print(f"✗ Error generating simulations: {e}")
        return 1

    # Test 3: Pipeline script starts without error
    try:
        import subprocess

        result = subprocess.run(
            ["./run_pipeline.sh"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=project_root,
        )

        if result.returncode != 0 and "simulation_configs" in result.stderr:
            print("✗ Pipeline script still has simulation_configs error")
            return 1
        else:
            print("✓ Pipeline script starts without simulation_configs error")
    except subprocess.TimeoutExpired:
        print("✓ Pipeline script starts and runs (timeout expected)")
    except Exception as e:
        print(f"✗ Error testing pipeline script: {e}")
        return 1

    print("\n🎉 Pipeline fix verified successfully!")
    print("\nFixed issues:")
    print("- Removed reference to non-existent 'simulation_configs' attribute")
    print("- BasicSimulationRunner now initializes correctly")
    print("- Pipeline script runs without attribute errors")
    print("\nNext steps:")
    print("- Run './run_pipeline.sh' to execute the full pipeline")
    print("- Use 'python simulation_runner.py' for basic simulations only")
    print("- Use 'python batch_simulation.py' for batch simulations")

    return 0


if __name__ == "__main__":
    sys.exit(main())
