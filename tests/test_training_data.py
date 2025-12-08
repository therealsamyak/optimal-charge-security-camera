#!/usr/bin/env python3
"""
Test Training Data Generation
Tests MIPS generation for 3 locations × 2 timestamps
"""

import sys
import logging
import json
import tempfile
from pathlib import Path

from src.energy_data import EnergyData
from src.power_profiler import PowerProfiler
from src.yolo_model import YOLOModel

# Setup logging for tests
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def test_energy_data_loading():
    """Test loading energy data for different locations."""
    logger.info("Testing energy data loading...")

    try:
        energy_data = EnergyData()
        logger.debug("EnergyData initialized")
    except Exception as e:
        logger.error(f"Failed to initialize EnergyData: {e}")
        raise

    # Test locations
    test_locations = [
        "US-CAL-LDWP_2024_5_minute",
        "US-FLA-FPL_2024_5_minute",
        "US-NY-NYIS_2024_5_minute",
    ]  # 3 locations as specified

    for location in test_locations:
        logger.debug(f"Testing location: {location}")
        try:
            # Test getting clean energy percentage at different times
            morning_percentage = energy_data.get_clean_energy_percentage(
                location, "2024-01-15 08:00:00"
            )
            evening_percentage = energy_data.get_clean_energy_percentage(
                location, "2024-01-15 20:00:00"
            )

            assert morning_percentage is not None, (
                f"Morning data missing for {location}"
            )
            assert evening_percentage is not None, (
                f"Evening data missing for {location}"
            )
            assert 0 <= morning_percentage <= 100, (
                f"Invalid morning percentage for {location}: {morning_percentage}"
            )
            assert 0 <= evening_percentage <= 100, (
                f"Invalid evening percentage for {location}: {evening_percentage}"
            )

            logger.debug(
                f"  ✓ {location}: Morning={morning_percentage:.1f}%, Evening={evening_percentage:.1f}%"
            )

        except Exception as e:
            logger.error(f"  ❌ {location}: Error - {e}")
            raise

    logger.info("✓ Energy data loading test passed")


def test_power_profiles():
    """Test loading power profiles for YOLO models."""
    print("Testing power profiles...")

    profiler = PowerProfiler()
    profiler.load_profiles()

    # Test all YOLOv10 models
    expected_models = [
        "YOLOv10_N",
        "YOLOv10_S",
        "YOLOv10_M",
        "YOLOv10_B",
        "YOLOv10_L",
        "YOLOv10_X",
    ]

    # Get all models data
    all_models = profiler.get_all_models_data()

    for model_name in expected_models:
        try:
            model_data = all_models.get(model_name)
            assert model_data is not None, f"No data for model {model_name}"

            # Check required fields
            required_fields = [
                "accuracy",
                "avg_inference_time_seconds",
                "model_power_mw",
            ]
            for field in required_fields:
                assert field in model_data, (
                    f"Missing field {field} for model {model_name}"
                )
                assert model_data[field] >= 0, (
                    f"Invalid {field} for model {model_name}: {model_data[field]}"
                )

            print(
                f"  ✓ {model_name}: accuracy={model_data['accuracy']:.3f}, "
                f"time={model_data['avg_inference_time_seconds']:.3f}s, "
                f"power={model_data['model_power_mw']:.3f}"
            )

        except Exception as e:
            print(f"  ❌ {model_name}: Error - {e}")
            raise

    print("✓ Power profiles test passed")


def test_yolo_model_loading():
    """Test YOLO model specifications loading."""
    print("Testing YOLO model loading...")

    # Test getting model specifications
    expected_models = [
        "YOLOv10_N",
        "YOLOv10_S",
        "YOLOv10_M",
        "YOLOv10_B",
        "YOLOv10_L",
        "YOLOv10_X",
    ]

    for model_name in expected_models:
        try:
            # Create YOLO model instance (version "n" for YOLOv10_N, etc.)
            version = model_name.split("_")[-1].lower()
            yolo_model = YOLOModel(model_name, version)
            specs = yolo_model.model_specs

            assert specs is not None, f"No specifications for model {model_name}"

            # Check required fields
            required_fields = ["latency_ms", "accuracy_map"]
            for field in required_fields:
                assert field in specs, f"Missing field {field} for model {model_name}"
                assert specs[field] >= 0, (
                    f"Invalid {field} for model {model_name}: {specs[field]}"
                )

            print(
                f"  ✓ {model_name}: accuracy={specs['accuracy_map']:.3f}, "
                f"latency={specs['latency_ms']:.1f}ms"
            )

        except Exception as e:
            print(f"  ❌ {model_name}: Error - {e}")
            raise

    print("✓ YOLO model loading test passed")


def test_mips_generation():
    """Test MIPS (Model Selection and Power Management) generation."""
    print("Testing MIPS generation...")

    # Initialize components
    energy_data = EnergyData()
    profiler = PowerProfiler()
    profiler.load_profiles()

    # Test scenarios: 3 locations × 2 timestamps
    locations = [
        "US-CAL-LDWP_2024_5_minute",
        "US-FLA-FPL_2024_5_minute",
        "US-NY-NYIS_2024_5_minute",
    ]
    timestamps = ["2024-01-15 08:00:00", "2024-01-15 20:00:00"]  # Morning and evening

    generated_scenarios = []

    for location in locations:
        for timestamp in timestamps:
            try:
                # Get environmental conditions
                clean_energy_pct = energy_data.get_clean_energy_percentage(
                    location, timestamp
                )
                assert clean_energy_pct is not None, (
                    f"No clean energy data for {location} at {timestamp}"
                )

                # Generate scenario with varying battery levels and requirements
                battery_levels = [20.0, 50.0, 80.0]  # Low, medium, high
                accuracy_requirements = [0.5, 0.8]  # Low, high
                latency_requirements = [10.0, 25.0]  # Fast, slow

                for battery_level in battery_levels:
                    for accuracy_req in accuracy_requirements:
                        for latency_req in latency_requirements:
                            # Determine optimal model based on requirements
                            available_models = profiler.get_all_models_data()
                            optimal_model = select_optimal_model(
                                available_models, accuracy_req, latency_req
                            )

                            # Determine charging decision
                            should_charge = should_charge_decision(
                                battery_level, clean_energy_pct
                            )

                            scenario = {
                                "location": location,
                                "timestamp": timestamp,
                                "battery_level": battery_level,
                                "clean_energy_percentage": clean_energy_pct,
                                "accuracy_requirement": accuracy_req,
                                "latency_requirement": latency_req,
                                "optimal_model": optimal_model,
                                "should_charge": should_charge,
                            }

                            generated_scenarios.append(scenario)

            except Exception as e:
                print(f"  ❌ {location} at {timestamp}: Error - {e}")
                raise

    # Validate generated scenarios
    assert len(generated_scenarios) == 3 * 2 * 3 * 2 * 2, (
        f"Expected 72 scenarios, got {len(generated_scenarios)}"
    )

    # Check scenario structure
    for scenario in generated_scenarios:
        required_fields = [
            "location",
            "timestamp",
            "battery_level",
            "clean_energy_percentage",
            "accuracy_requirement",
            "latency_requirement",
            "optimal_model",
            "should_charge",
        ]
        for field in required_fields:
            assert field in scenario, f"Missing field {field} in scenario"

    # Check value ranges
    for scenario in generated_scenarios:
        assert 0 <= scenario["battery_level"] <= 100, (
            f"Invalid battery level: {scenario['battery_level']}"
        )
        assert 0 <= scenario["clean_energy_percentage"] <= 100, (
            f"Invalid clean energy: {scenario['clean_energy_percentage']}"
        )
        assert 0 <= scenario["accuracy_requirement"] <= 1, (
            f"Invalid accuracy requirement: {scenario['accuracy_requirement']}"
        )
        assert scenario["latency_requirement"] > 0, (
            f"Invalid latency requirement: {scenario['latency_requirement']}"
        )
        assert scenario["optimal_model"] in [
            "YOLOv10_N",
            "YOLOv10_S",
            "YOLOv10_M",
            "YOLOv10_B",
            "YOLOv10_L",
            "YOLOv10_X",
        ], f"Invalid model: {scenario['optimal_model']}"
        assert isinstance(scenario["should_charge"], bool), (
            f"Invalid charge decision: {scenario['should_charge']}"
        )

    print(
        f"✓ MIPS generation test passed ({len(generated_scenarios)} scenarios generated)"
    )


def generate_test_scenarios():
    """Generate test scenarios for other functions."""
    # Initialize components
    energy_data = EnergyData()
    profiler = PowerProfiler()
    profiler.load_profiles()

    # Test scenarios: 3 locations × 2 timestamps
    locations = [
        "US-CAL-LDWP_2024_5_minute",
        "US-FLA-FPL_2024_5_minute",
        "US-NY-NYIS_2024_5_minute",
    ]
    timestamps = ["2024-01-15 08:00:00", "2024-01-15 20:00:00"]  # Morning and evening

    generated_scenarios = []

    for location in locations:
        for timestamp in timestamps:
            try:
                # Get environmental conditions
                clean_energy_pct = energy_data.get_clean_energy_percentage(
                    location, timestamp
                )
                assert clean_energy_pct is not None, (
                    f"No clean energy data for {location} at {timestamp}"
                )

                # Generate scenario with varying battery levels and requirements
                battery_levels = [20.0, 50.0, 80.0]  # Low, medium, high
                accuracy_requirements = [0.5, 0.8]  # Low, high
                latency_requirements = [10.0, 25.0]  # Fast, slow

                for battery_level in battery_levels:
                    for accuracy_req in accuracy_requirements:
                        for latency_req in latency_requirements:
                            # Determine optimal model based on requirements
                            available_models = profiler.get_all_models_data()
                            optimal_model = select_optimal_model(
                                available_models, accuracy_req, latency_req
                            )

                            # Determine charging decision
                            should_charge = should_charge_decision(
                                battery_level, clean_energy_pct
                            )

                            scenario = {
                                "location": location,
                                "timestamp": timestamp,
                                "battery_level": battery_level,
                                "clean_energy_percentage": clean_energy_pct,
                                "accuracy_requirement": accuracy_req,
                                "latency_requirement": latency_req,
                                "optimal_model": optimal_model,
                                "should_charge": should_charge,
                            }

                            generated_scenarios.append(scenario)

            except Exception as e:
                print(f"  ❌ {location} at {timestamp}: Error - {e}")
                raise

    return generated_scenarios


def select_optimal_model(available_models, accuracy_requirement, latency_requirement):
    """Select optimal model based on requirements."""
    best_model = None
    best_score = -1

    for model_name, model_data in available_models.items():
        # Check if model meets requirements
        if (
            model_data["accuracy"] >= accuracy_requirement
            and model_data["avg_inference_time_seconds"] * 1000 <= latency_requirement
        ):
            # Score based on efficiency (lower power is better)
            score = 1.0 / (
                model_data["model_power_mw"] + 0.001
            )  # Avoid division by zero
            if score > best_score:
                best_score = score
                best_model = model_name

    # If no model meets requirements, choose the most efficient one
    if best_model is None:
        best_model = min(
            available_models.keys(), key=lambda x: available_models[x]["model_power_mw"]
        )

    return best_model


def should_charge_decision(battery_level, clean_energy_percentage):
    """Determine if should charge based on battery and clean energy."""
    # Charge if battery is low or clean energy is high
    return battery_level < 30 or (battery_level < 70 and clean_energy_percentage > 70)


def test_training_data_export():
    """Test exporting training data to JSON."""
    logger.info("Testing training data export...")

    # Generate test scenarios
    scenarios = generate_test_scenarios()
    logger.debug(f"Generated {len(scenarios)} scenarios for export")

    # Use temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        output_file = temp_path / "test_training_data.json"
        logger.debug(f"Using temporary file: {output_file}")

        try:
            with open(output_file, "w") as f:
                json.dump(scenarios, f, indent=2)
            logger.debug("Scenarios exported to JSON")

            # Verify export
            assert output_file.exists(), "Training data file not created"

            # Load and verify
            with open(output_file, "r") as f:
                loaded_scenarios = json.load(f)
            logger.debug("Scenarios loaded back from JSON")

            assert len(loaded_scenarios) == len(scenarios), (
                "Exported data length mismatch"
            )
            assert loaded_scenarios == scenarios, "Exported data mismatch"

            logger.info(
                f"✓ Training data export test passed ({len(scenarios)} scenarios exported)"
            )

        except Exception as e:
            logger.error(f"  ❌ Export error: {e}")
            raise


def test_data_quality():
    """Test data quality and consistency."""
    print("Testing data quality...")

    # Generate scenarios
    scenarios = generate_test_scenarios()

    # Test for duplicates
    unique_scenarios = []
    for scenario in scenarios or []:
        scenario_key = (
            scenario["location"],
            scenario["timestamp"],
            scenario["battery_level"],
            scenario["accuracy_requirement"],
            scenario["latency_requirement"],
        )
        if scenario_key not in unique_scenarios:
            unique_scenarios.append(scenario_key)
        else:
            assert False, f"Duplicate scenario found: {scenario_key}"

    # Test distribution
    location_counts = {}
    model_counts = {}
    charge_decisions = {"charge": 0, "no_charge": 0}

    for scenario in scenarios:
        # Location distribution
        if scenarios:
            location_counts[scenario["location"]] = (
                location_counts.get(scenario["location"], 0) + 1
            )

        # Model distribution
        if scenarios:
            model_counts[scenario["optimal_model"]] = (
                model_counts.get(scenario["optimal_model"], 0) + 1
            )

        # Charge decision distribution
        if scenarios and scenario["should_charge"]:
            charge_decisions["charge"] += 1
        else:
            charge_decisions["no_charge"] += 1

    # Verify distributions are reasonable
    assert len(location_counts) == 3, (
        f"Expected 3 locations, got {len(location_counts)}"
    )
    assert all(count > 0 for count in location_counts.values()), (
        "Some locations have no scenarios"
    )

    assert len(model_counts) > 1, "Expected multiple models to be selected"
    assert all(count > 0 for count in model_counts.values()), (
        "Some models have no scenarios"
    )

    total_scenarios = len(scenarios) if scenarios else 0
    charge_ratio = (
        charge_decisions["charge"] / total_scenarios if total_scenarios > 0 else 0
    )
    assert 0.1 <= charge_ratio <= 0.9, f"Unreasonable charge ratio: {charge_ratio:.2f}"

    print("✓ Data quality test passed")
    if scenarios:
        print(f"  Location distribution: {location_counts}")
        print(f"  Model distribution: {model_counts}")
        print(f"  Charge decisions: {charge_decisions}")


def main():
    """Run all training data tests."""
    logger.info("🧪 Running Training Data Tests")
    logger.info("=" * 50)

    try:
        test_energy_data_loading()
        test_power_profiles()
        test_yolo_model_loading()
        test_mips_generation()
        test_training_data_export()
        test_data_quality()

        logger.info("\n✅ All training data tests passed!")
        return 0

    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
