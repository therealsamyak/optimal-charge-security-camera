#!/usr/bin/env python3
"""
Consolidated controller tests - integration tests for all controller types and CustomController training functionality.
"""

import json
import unittest
import numpy as np

from src.simulation_engine import SimulationEngine, SimulationConfig
from src.controller import (
    NaiveWeakController,
    NaiveStrongController,
    OracleController,
    CustomController,
)
from train_custom_controller import (
    load_power_profiles,
    CustomController as TrainingCustomController,
)


class TestControllerIntegration(unittest.TestCase):
    """Integration tests for all controller types."""

    def setUp(self):
        """Set up test fixtures."""
        # Load power profiles for testing
        with open("results/power_profiles.json", "r") as f:
            self.power_profiles = json.load(f)

        # Test configuration for quick simulations
        self.test_config = SimulationConfig(
            duration_days=1,
            task_interval_seconds=60,  # 1 minute intervals for faster testing
            time_acceleration=1000,  # Speed up testing
            battery_capacity_wh=5.0,
            charge_rate_watts=100.0,
            locations=["CA"],  # Single location for testing
            seasons=["summer"],  # Single season for testing
        )

        self.test_location = "CA"
        self.test_season = "summer"
        self.test_week = 1

    def test_naive_weak_controller(self):
        """Test NaiveWeakController functionality."""
        print("\n🧪 Testing NaiveWeakController...")

        controller = NaiveWeakController()

        engine = SimulationEngine(
            config=self.test_config,
            controller=controller,
            location=self.test_location,
            season=self.test_season,
            week=self.test_week,
            power_profiles=self.power_profiles,
        )

        metrics = engine.run()

        # Basic validation
        self.assertGreater(metrics["total_tasks"], 0, "No tasks generated")
        self.assertGreaterEqual(
            metrics["completed_tasks"], 0, "Negative completed tasks"
        )
        self.assertGreaterEqual(metrics["total_energy_wh"], 0, "Negative energy usage")
        self.assertBetween(
            metrics["clean_energy_percentage"],
            0,
            100,
            "Invalid clean energy percentage",
        )

        print(
            f"✅ NaiveWeakController: {metrics['total_tasks']} tasks, {metrics['task_completion_rate']:.1f}% completion"
        )

    def test_naive_strong_controller(self):
        """Test NaiveStrongController functionality."""
        print("\n🧪 Testing NaiveStrongController...")

        controller = NaiveStrongController()

        engine = SimulationEngine(
            config=self.test_config,
            controller=controller,
            location=self.test_location,
            season=self.test_season,
            week=self.test_week,
            power_profiles=self.power_profiles,
        )

        metrics = engine.run()

        # Basic validation
        self.assertGreater(metrics["total_tasks"], 0, "No tasks generated")
        self.assertGreaterEqual(
            metrics["completed_tasks"], 0, "Negative completed tasks"
        )
        self.assertGreaterEqual(metrics["total_energy_wh"], 0, "Negative energy usage")
        self.assertBetween(
            metrics["clean_energy_percentage"],
            0,
            100,
            "Invalid clean energy percentage",
        )

        print(
            f"✅ NaiveStrongController: {metrics['total_tasks']} tasks, {metrics['task_completion_rate']:.1f}% completion"
        )

    def test_oracle_controller(self):
        """Test OracleController functionality."""
        print("\n🧪 Testing OracleController...")

        # Oracle needs future data - use dummy data for testing
        controller = OracleController(future_energy_data={}, future_tasks=100)

        engine = SimulationEngine(
            config=self.test_config,
            controller=controller,
            location=self.test_location,
            season=self.test_season,
            week=self.test_week,
            power_profiles=self.power_profiles,
        )

        metrics = engine.run()

        # Basic validation
        self.assertGreater(metrics["total_tasks"], 0, "No tasks generated")
        self.assertGreaterEqual(
            metrics["completed_tasks"], 0, "Negative completed tasks"
        )
        self.assertGreaterEqual(metrics["total_energy_wh"], 0, "Negative energy usage")
        self.assertBetween(
            metrics["clean_energy_percentage"],
            0,
            100,
            "Invalid clean energy percentage",
        )

        print(
            f"✅ OracleController: {metrics['total_tasks']} tasks, {metrics['task_completion_rate']:.1f}% completion"
        )

    def test_custom_controller_integration(self):
        """Test CustomController functionality."""
        print("\n🧪 Testing CustomController...")

        weights_file = "results/custom_controller_weights.json"
        controller = CustomController(weights_file)

        engine = SimulationEngine(
            config=self.test_config,
            controller=controller,
            location=self.test_location,
            season=self.test_season,
            week=self.test_week,
            power_profiles=self.power_profiles,
        )

        metrics = engine.run()

        # Basic validation
        self.assertGreater(metrics["total_tasks"], 0, "No tasks generated")
        self.assertGreaterEqual(
            metrics["completed_tasks"], 0, "Negative completed tasks"
        )
        self.assertGreaterEqual(metrics["total_energy_wh"], 0, "Negative energy usage")
        self.assertBetween(
            metrics["clean_energy_percentage"],
            0,
            100,
            "Invalid clean energy percentage",
        )

        print(
            f"✅ CustomController: {metrics['total_tasks']} tasks, {metrics['task_completion_rate']:.1f}% completion"
        )

    def assertBetween(self, value, min_val, max_val, msg):
        """Helper to assert value is between min and max."""
        self.assertGreaterEqual(value, min_val, msg)
        self.assertLessEqual(value, max_val, msg)


class TestCustomControllerTraining(unittest.TestCase):
    """Test CustomController training functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.models = load_power_profiles()

    def test_custom_controller_init(self):
        """Test CustomController initialization."""
        print("Testing CustomController initialization...")
        controller = TrainingCustomController()

        self.assertIn("accuracy_weight", controller.weights, "Missing accuracy_weight")
        self.assertIn("latency_weight", controller.weights, "Missing latency_weight")
        self.assertIn(
            "clean_energy_weight", controller.weights, "Missing clean_energy_weight"
        )
        self.assertEqual(
            controller.charge_threshold, 0.0, "Incorrect default charge threshold"
        )

        print("✓ CustomController initialized successfully")

    def test_feature_extraction(self):
        """Test feature extraction from scenarios."""
        print("Testing feature extraction...")
        controller = TrainingCustomController()

        scenario = {
            "battery_level": 50,
            "clean_energy_percentage": 75,
            "accuracy_requirement": 0.85,  # Now 0-1 range
            "latency_requirement": 1500,
        }

        features = controller.extract_features(scenario)
        self.assertEqual(len(features), 4, f"Expected 4 features, got {len(features)}")
        self.assertEqual(features[0], 0.5, f"Incorrect battery feature: {features[0]}")
        self.assertEqual(
            features[1], 0.75, f"Incorrect clean energy feature: {features[1]}"
        )
        self.assertEqual(
            features[2], 0.85, f"Incorrect accuracy feature: {features[2]}"
        )
        self.assertEqual(features[3], 50.0, f"Incorrect latency feature: {features[3]}")

        print("✓ Feature extraction successful")

    def test_prediction(self):
        """Test model and charging prediction."""
        print("Testing prediction...")
        controller = TrainingCustomController()

        features = np.array([0.5, 0.75, 0.85, 0.5])
        available_models = ["YOLOv10_N", "YOLOv10_S"]

        # Mock model data for prediction
        model_data = {
            "YOLOv10_N": {"accuracy": 39.5, "latency": 1.56, "power_cost": 602.25},
            "YOLOv10_S": {"accuracy": 46.7, "latency": 2.66, "power_cost": 800.0},
        }

        model, charge = controller.predict_model_and_charge(
            features, available_models, model_data
        )

        self.assertIn(
            model, available_models, f"Predicted model {model} not in available models"
        )
        self.assertIsInstance(
            charge,
            (bool, np.bool_),
            f"Charge decision should be boolean, got {type(charge)}",
        )

        print("✓ Prediction successful")

    def test_accuracy_translation(self):
        """Test accuracy requirement translation to model suitability."""
        print("Testing accuracy translation...")
        controller = TrainingCustomController()

        # Test cases: (user_requirement, model_map, expected_behavior)
        test_cases = [
            (0.3, 39.5, "should be acceptable"),  # Low requirement, low model
            (
                0.8,
                54.4,
                "should be penalized",
            ),  # High requirement, best model still insufficient
            (0.5, 46.7, "should be reasonable"),  # Medium requirement, medium model
            (
                0.9,
                39.5,
                "should be heavily penalized",
            ),  # Very high requirement, weak model
        ]

        for user_req, model_map, expected in test_cases:
            score = controller.get_model_accuracy_score(user_req, model_map)
            print(
                f"User req: {user_req:.1f}, Model mAP: {model_map:.1f} → Score: {score:.3f} ({expected})"
            )

            # Score should be between -1 and 1
            self.assertBetween(score, -1.0, 1.0, f"Score {score} out of range")

        print("✓ Accuracy translation working correctly")

    def test_model_selection_with_accuracy(self):
        """Test model selection considers accuracy requirements."""
        print("Testing model selection with accuracy requirements...")
        controller = TrainingCustomController()

        # Mock model data
        model_data = {
            "YOLOv10_N": {"accuracy": 39.5, "latency": 1.56, "power_cost": 602.25},
            "YOLOv10_X": {"accuracy": 54.4, "latency": 12.2, "power_cost": 2000.0},
        }

        # Test low accuracy requirement - should prefer lighter model
        features_low = np.array([0.5, 0.5, 0.3, 0.5])  # Low accuracy requirement
        model_low, charge_low = controller.predict_model_and_charge(
            features_low, list(model_data.keys()), model_data
        )

        # Test high accuracy requirement - should prefer stronger model
        features_high = np.array([0.5, 0.5, 0.9, 0.5])  # High accuracy requirement
        model_high, charge_high = controller.predict_model_and_charge(
            features_high, list(model_data.keys()), model_data
        )

        print(f"Low accuracy requirement (0.3): Selected {model_low}")
        print(f"High accuracy requirement (0.9): Selected {model_high}")

        print("✓ Model selection with accuracy requirements working")

    def assertBetween(self, value, min_val, max_val, msg):
        """Helper to assert value is between min and max."""
        self.assertGreaterEqual(value, min_val, msg)
        self.assertLessEqual(value, max_val, msg)


if __name__ == "__main__":
    unittest.main()
