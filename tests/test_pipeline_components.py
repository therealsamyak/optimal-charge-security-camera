#!/usr/bin/env python3
"""
Test pipeline components without file I/O
"""

import sys
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))


def test_training_data_components():
    """Test training data generation components."""
    try:
        from generate_training_data import load_energy_data, generate_training_scenarios

        print("Testing training data components...")

        # Test energy data loading
        energy_data = load_energy_data()
        print(f"✓ Loaded energy data for {len(energy_data)} regions")

        # Test scenario generation (small sample) - just test that it runs
        scenarios = generate_training_scenarios(energy_data)[:5]  # Take first 5
        print(f"✓ Generated {len(scenarios)} training scenarios")

        # Check scenario structure
        if scenarios:
            scenario = scenarios[0]
            print(f"✓ Sample scenario: {scenario}")

        return True

    except Exception as e:
        print(f"✗ Error in training data components: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_controller_components():
    """Test controller training components."""
    try:
        from train_custom_controller import CustomController

        print("Testing controller components...")

        # Create controller
        controller = CustomController()
        print("✓ Controller initialized")

        # Test feature extraction with dummy data
        dummy_scenario = {
            "battery_level": 0.5,
            "energy_price": 0.15,
            "clean_energy_percentage": 0.3,
            "user_accuracy_requirement": 0.8,
            "user_latency_requirement": 5.0,
            "tasks_pending": 1,
        }

        features = controller.extract_features(dummy_scenario)
        print(f"✓ Feature extraction: {features.shape}")

        # Test model scoring
        score = controller.get_model_accuracy_score(0.8, 0.9)
        print(f"✓ Model scoring: {score}")

        return True

    except Exception as e:
        print(f"✗ Error in controller components: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_config_loading():
    """Test configuration loading."""
    try:
        from src.config_loader import ConfigLoader

        print("Testing configuration loading...")

        config_loader = ConfigLoader("config.jsonc")
        print("✓ Config loaded")

        # Test various config methods
        locations = config_loader.get_locations()
        seasons = config_loader.get_seasons()
        controllers = config_loader.get_controllers()

        print(f"✓ Locations: {locations}")
        print(f"✓ Seasons: {seasons}")
        print(f"✓ Controllers: {controllers}")

        sim_config = config_loader.get_simulation_config()
        print(f"✓ Simulation config: {sim_config}")

        return True

    except Exception as e:
        print(f"✗ Error in config loading: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🧪 Testing Pipeline Components (No File I/O)")
    print("=" * 50)

    tests = [
        ("Config Loading", test_config_loading),
        ("Training Data Components", test_training_data_components),
        ("Controller Components", test_controller_components),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}:")
        success = test_func()
        results.append((test_name, success))

    print("\n" + "=" * 50)
    print("📊 Test Results:")
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {test_name}: {status}")

    all_passed = all(success for _, success in results)
    print(f"\n{'✅ All tests passed!' if all_passed else '❌ Some tests failed!'}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
