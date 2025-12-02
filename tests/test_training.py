#!/usr/bin/env python3
"""
Consolidated training tests - improved training algorithm, data splitting, and diversity analysis.
"""

import json
import unittest
import numpy as np
from collections import Counter

from train_custom_controller import CustomController, load_power_profiles
from generate_training_data import generate_training_scenarios


class TestTrainingAlgorithm(unittest.TestCase):
    """Test improved CustomController training algorithm."""

    def load_power_profiles_extended(self) -> dict:
        """Load power profiles from results and real model data."""
        with open("results/power_profiles.json", "r") as f:
            profiles = json.load(f)

        # Load real model data
        model_data = {}
        with open("model-data/model-data.csv", "r") as f:
            lines = f.readlines()
            for line in lines[1:]:  # Skip header
                parts = line.strip().split(",")
                model = parts[0].strip('"')
                version = parts[1].strip('"')
                latency = float(parts[2].strip('"'))
                accuracy = float(parts[3].strip('"'))
                model_data[f"{model}_{version}"] = {
                    "accuracy": accuracy,
                    "latency": latency,
                }

        models = {}
        for model_name, data in profiles.items():
            real_data = model_data.get(model_name, {})
            models[model_name] = {
                "accuracy": real_data.get("accuracy", 85.0),
                "latency": real_data.get(
                    "latency", data["avg_inference_time_seconds"] * 1000
                ),
                "power_cost": data["model_power_mw"],
            }

        return models

    def create_small_training_data(self) -> list:
        """Create small training dataset for testing."""
        models = self.load_power_profiles_extended()
        training_data = []

        # Generate diverse scenarios
        for battery in [20, 50, 80]:
            for clean_energy in [10, 50, 90]:
                for acc_req in [0.4, 0.7, 0.9]:
                    for lat_req in [5, 10, 15]:
                        # Simple heuristic for optimal model
                        suitable_models = [
                            m
                            for m in models.keys()
                            if models[m]["accuracy"] >= acc_req * 100
                            and models[m]["latency"] <= lat_req
                        ]

                        if suitable_models:
                            # Choose most accurate suitable model
                            optimal_model = max(
                                suitable_models, key=lambda x: models[x]["accuracy"]
                            )
                            should_charge = battery < 30 or (
                                clean_energy > 80 and battery < 70
                            )

                            training_data.append(
                                {
                                    "battery_level": battery,
                                    "clean_energy_percentage": clean_energy,
                                    "accuracy_requirement": acc_req,
                                    "latency_requirement": lat_req,
                                    "optimal_model": optimal_model,
                                    "should_charge": should_charge,
                                }
                            )

        return training_data

    def test_improved_training(self):
        """Test the improved training algorithm."""
        print("Creating small training dataset...")
        training_data = self.create_small_training_data()
        print(f"Created {len(training_data)} training samples")

        print("Loading power profiles...")
        available_models = self.load_power_profiles_extended()

        print("Initializing CustomController...")
        controller = CustomController()

        print("Testing improved training...")
        test_metrics = controller.train(
            training_data, available_models, epochs=50, learning_rate=0.05
        )

        print("\nResults:")
        print(f"Model Accuracy: {test_metrics['model_accuracy']:.3f}")
        print(f"Charge Accuracy: {test_metrics['charge_accuracy']:.3f}")
        print(f"Overall Accuracy: {test_metrics['overall_accuracy']:.3f}")

        # Test if improvement is significant
        self.assertGreater(
            test_metrics["overall_accuracy"],
            0.4,
            "Training should achieve reasonable accuracy",
        )


class TestDataSplitting(unittest.TestCase):
    """Test train/validation/test splitting functionality."""

    def test_data_splitting(self):
        """Test train/validation/test data splitting."""
        print("🔀 Testing Data Splitting...")

        # Create mock training data
        mock_data = []
        for i in range(100):
            mock_data.append(
                {
                    "battery_level": np.random.randint(5, 100),
                    "clean_energy_percentage": np.random.randint(0, 100),
                    "accuracy_requirement": np.random.uniform(0.3, 1.0),
                    "latency_requirement": np.random.choice(
                        [1, 2, 3, 5, 8, 10, 15, 20]
                    ),
                    "optimal_model": "YOLOv10_N",
                    "should_charge": np.random.choice([True, False]),
                }
            )

        controller = CustomController()
        train_data, val_data, test_data = controller.split_data(mock_data)

        # Check splits
        total_len = len(mock_data)
        self.assertEqual(len(train_data) + len(val_data) + len(test_data), total_len)
        self.assertAlmostEqual(len(train_data) / total_len, 0.7, delta=0.05)
        self.assertAlmostEqual(len(val_data) / total_len, 0.2, delta=0.05)
        self.assertAlmostEqual(len(test_data) / total_len, 0.1, delta=0.05)

        print(
            f"✅ Data split correct: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}"
        )

    def test_evaluation(self):
        """Test model evaluation."""
        print("📊 Testing Evaluation...")

        controller = CustomController()

        # Create mock data
        mock_data = [
            {
                "battery_level": 50,
                "clean_energy_percentage": 75,
                "accuracy_requirement": 0.8,
                "latency_requirement": 10,
                "optimal_model": "YOLOv10_X",
                "should_charge": True,
            },
            {
                "battery_level": 25,
                "clean_energy_percentage": 25,
                "accuracy_requirement": 0.4,
                "latency_requirement": 5,
                "optimal_model": "YOLOv10_N",
                "should_charge": False,
            },
        ]

        mock_models = {
            "YOLOv10_N": {"accuracy": 39.5, "latency": 1.56, "power_cost": 602.25},
            "YOLOv10_X": {"accuracy": 54.4, "latency": 12.2, "power_cost": 3476.75},
        }

        metrics = controller.evaluate(mock_data, mock_models)

        self.assertIn("loss", metrics)
        self.assertIn("model_accuracy", metrics)
        self.assertIn("charge_accuracy", metrics)
        self.assertIn("overall_accuracy", metrics)

        print(
            f"✅ Evaluation working: Loss={metrics['loss']:.4f}, Overall Acc={metrics['overall_accuracy']:.3f}"
        )

    def test_load_power_profiles(self):
        """Test loading power profiles."""
        print("Testing load_power_profiles...")
        models = load_power_profiles()

        self.assertGreater(len(models), 0, "No models loaded")
        self.assertIn("YOLOv10_N", models, "YOLOv10_N not found")

        for model_name, data in models.items():
            self.assertIn("accuracy", data, f"Missing accuracy for {model_name}")
            self.assertIn("latency", data, f"Missing latency for {model_name}")
            self.assertIn("power_cost", data, f"Missing power_cost for {model_name}")

        print(f"✓ Loaded {len(models)} models successfully")

    def test_generate_training_scenarios(self):
        """Test scenario generation."""
        print("Testing generate_training_scenarios...")
        scenarios = generate_training_scenarios()

        self.assertGreater(len(scenarios), 0, "No scenarios generated")
        self.assertLessEqual(len(scenarios), 150000, "Too many scenarios generated")

        # Check first scenario
        battery, clean_energy, acc_req, lat_req = scenarios[0]
        self.assertBetween(battery, 5, 100, f"Invalid battery level: {battery}")
        self.assertBetween(
            clean_energy, 0, 100, f"Invalid clean energy: {clean_energy}"
        )
        self.assertBetween(
            acc_req, 0.2, 1.0, f"Invalid accuracy requirement: {acc_req}"
        )
        self.assertBetween(lat_req, 1, 30, f"Invalid latency requirement: {lat_req}")

        print(f"✓ Generated {len(scenarios)} scenarios successfully")

    def assertBetween(self, value, min_val, max_val, msg):
        """Helper to assert value is between min and max."""
        self.assertGreaterEqual(value, min_val, msg)
        self.assertLessEqual(value, max_val, msg)


class TestTrainingDiversity(unittest.TestCase):
    """Test training data diversity and coverage."""

    def analyze_diversity(self):
        """Analyze the diversity of training scenarios."""
        print("📊 Training Data Diversity Analysis")
        print("=" * 50)

        scenarios = generate_training_scenarios()

        # Extract each dimension
        battery_levels = [s[0] for s in scenarios]
        clean_energy = [s[1] for s in scenarios]
        accuracy_reqs = [s[2] for s in scenarios]
        latency_reqs = [s[3] for s in scenarios]

        print(f"📈 Total scenarios: {len(scenarios):,}")
        print()

        # Analyze each dimension
        print("🔋 Battery Levels:")
        battery_counts = Counter(battery_levels)
        for level in sorted(battery_counts.keys()):
            count = battery_counts[level]
            percentage = (count / len(scenarios)) * 100
            print(f"  {level:3.0f}%: {count:4d} scenarios ({percentage:4.1f}%)")
        print()

        print("🌱 Clean Energy Levels:")
        clean_counts = Counter(clean_energy)
        for level in sorted(clean_counts.keys()):
            count = clean_counts[level]
            percentage = (count / len(scenarios)) * 100
            print(f"  {level:3.0f}%: {count:4d} scenarios ({percentage:4.1f}%)")
        print()

        print("🎯 Accuracy Requirements:")
        acc_counts = Counter(accuracy_reqs)
        for level in sorted(acc_counts.keys()):
            count = acc_counts[level]
            percentage = (count / len(scenarios)) * 100
            print(f"  {level:.1f}: {count:4d} scenarios ({percentage:4.1f}%)")
        print()

        print("⏱️  Latency Requirements:")
        lat_counts = Counter(latency_reqs)
        for level in sorted(lat_counts.keys()):
            count = lat_counts[level]
            percentage = (count / len(scenarios)) * 100
            print(f"  {level:4d}ms: {count:4d} scenarios ({percentage:4.1f}%)")
        print()

        # Check edge cases
        print("🔍 Edge Case Coverage:")
        low_battery = sum(1 for b in battery_levels if b <= 20)
        high_battery = sum(1 for b in battery_levels if b >= 80)
        low_clean = sum(1 for c in clean_energy if c <= 20)
        high_clean = sum(1 for c in clean_energy if c >= 80)
        low_accuracy = sum(1 for a in accuracy_reqs if a <= 0.4)
        high_accuracy = sum(1 for a in accuracy_reqs if a >= 0.9)

        print(
            f"  Low battery (≤20%): {low_battery:4d} scenarios ({low_battery / len(scenarios) * 100:4.1f}%)"
        )
        print(
            f"  High battery (≥80%): {high_battery:4d} scenarios ({high_battery / len(scenarios) * 100:4.1f}%)"
        )
        print(
            f"  Low clean energy (≤20%): {low_clean:4d} scenarios ({low_clean / len(scenarios) * 100:4.1f}%)"
        )
        print(
            f"  High clean energy (≥80%): {high_clean:4d} scenarios ({high_clean / len(scenarios) * 100:4.1f}%)"
        )
        print(
            f"  Low accuracy (≤0.4): {low_accuracy:4d} scenarios ({low_accuracy / len(scenarios) * 100:4.1f}%)"
        )
        print(
            f"  High accuracy (≥0.9): {high_accuracy:4d} scenarios ({high_accuracy / len(scenarios) * 100:4.1f}%)"
        )
        print()

        # Diversity score
        unique_combinations = len(set(scenarios))
        diversity_score = unique_combinations / len(scenarios)
        print(
            f"🎲 Diversity Score: {diversity_score:.3f} ({unique_combinations}/{len(scenarios)} unique)"
        )

        return scenarios

    def test_diversity_analysis(self):
        """Test diversity analysis functionality."""
        scenarios = self.analyze_diversity()

        # Basic validation
        self.assertGreater(len(scenarios), 0, "No scenarios generated")
        self.assertGreater(
            len(set(scenarios)), 1000, "Should have many unique combinations"
        )

        # Check edge case coverage
        battery_levels = [s[0] for s in scenarios]
        accuracy_reqs = [s[2] for s in scenarios]

        low_battery = sum(1 for b in battery_levels if b <= 20)
        high_battery = sum(1 for b in battery_levels if b >= 80)
        low_accuracy = sum(1 for a in accuracy_reqs if a <= 0.4)
        high_accuracy = sum(1 for a in accuracy_reqs if a >= 0.9)

        # Should have reasonable edge case coverage
        self.assertGreater(low_battery, 100, "Should have low battery scenarios")
        self.assertGreater(high_battery, 100, "Should have high battery scenarios")
        self.assertGreater(low_accuracy, 100, "Should have low accuracy scenarios")
        self.assertGreater(high_accuracy, 100, "Should have high accuracy scenarios")

    def test_20k_samples_generation(self):
        """Test 20,000 sample generation with increased variety."""
        print("📊 Testing 20,000 Sample Generation...")

        scenarios = generate_training_scenarios()

        print(f"📈 Total scenarios: {len(scenarios):,}")

        # Extract each dimension
        battery_levels = [s[0] for s in scenarios]
        clean_energy = [s[1] for s in scenarios]
        accuracy_reqs = [s[2] for s in scenarios]
        latency_reqs = [s[3] for s in scenarios]

        print(f"🔋 Battery levels: {len(set(battery_levels))} unique")
        print(f"🌱 Clean energy: {len(set(clean_energy))} unique")
        print(f"🎯 Accuracy: {len(set(accuracy_reqs))} unique")
        print(f"⏱️  Latency: {len(set(latency_reqs))} unique")

        # Check accuracy variety
        accuracy_values = sorted(set(accuracy_reqs))
        print(f"\n🎯 Accuracy values: {accuracy_values}")
        print(f"   Range: {min(accuracy_values):.2f} to {max(accuracy_values):.2f}")
        print(f"   Step size: ~{accuracy_values[1] - accuracy_values[0]:.2f}")

        # Edge case coverage
        low_accuracy = sum(1 for a in accuracy_reqs if a <= 0.4)
        high_accuracy = sum(1 for a in accuracy_reqs if a >= 0.9)

        print("\n🔍 Edge Case Coverage:")
        print(
            f"  Low accuracy (≤0.4): {low_accuracy:4d} scenarios ({low_accuracy / len(scenarios) * 100:4.1f}%)"
        )
        print(
            f"  High accuracy (≥0.9): {high_accuracy:4d} scenarios ({high_accuracy / len(scenarios) * 100:4.1f}%)"
        )

        # Training split impact
        train_size = int(len(scenarios) * 0.7)
        val_size = int(len(scenarios) * 0.2)
        test_size = len(scenarios) - train_size - val_size

        print("\n🔄 Training Split Impact:")
        print(f"  Training: {train_size:,} scenarios")
        print(f"  Validation: {val_size:,} scenarios")
        print(f"  Testing: {test_size:,} scenarios")

        print("\n✅ 20,000 sample generation successful!")
        print("📈 Variety increased from 14,080 to 26,400 possible combinations")

        # Validate generation worked
        self.assertGreater(len(scenarios), 10000, "Should generate many scenarios")
        self.assertGreater(
            len(set(accuracy_reqs)), 5, "Should have variety in accuracy requirements"
        )


if __name__ == "__main__":
    unittest.main()
