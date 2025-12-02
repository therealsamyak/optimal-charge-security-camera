#!/usr/bin/env python3
"""
CustomController training using gradient descent.
Trains model selection and charging decisions using MIPS-generated training data.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np


class CustomController:
    """Custom controller with trained weights for model selection and charging."""

    def __init__(self):
        self.weights = {
            "accuracy_weight": 0.5,  # α
            "latency_weight": 0.3,  # β
            "clean_energy_weight": 0.2,  # γ
        }
        self.model_weights = {}
        self.charge_weights = None  # Learnable weights for charge decision
        self.charge_threshold = 0.0  # Threshold for charging decision
        self.logger = logging.getLogger(__name__)

    def load_training_data(self, filepath: str) -> List[Dict]:
        """Load training data from JSON file."""
        with open(filepath, "r") as f:
            return json.load(f)

    def get_model_accuracy_score(
        self, user_requirement: float, model_map: float
    ) -> float:
        """Convert user 0-1 requirement to model suitability score."""
        # Model accuracy is already in 0-1 range
        normalized_model_score = model_map
        normalized_model_score = np.clip(normalized_model_score, 0, 1)

        # If user requirement > model capability, penalize
        if user_requirement > normalized_model_score:
            penalty = (user_requirement - normalized_model_score) * 0.5
            return max(-1.0, normalized_model_score - penalty)

        return normalized_model_score

    def extract_features(self, scenario: Dict) -> np.ndarray:
        """Extract features from training scenario."""
        return np.array(
            [
                scenario["battery_level"] / 100.0,
                scenario["clean_energy_percentage"] / 100.0,
                scenario["accuracy_requirement"],  # Already 0-1 range
                scenario["latency_requirement"] / 30.0,  # Normalize to 30ms max
            ]
        )

    def predict_model_and_charge(
        self,
        features: np.ndarray,
        available_models: List[str],
        model_data: Dict[str, Dict[str, float]],
    ) -> Tuple[str, bool]:
        """Predict model selection and charging decision using current weights."""
        features[2]

        # Initialize weights if needed
        if self.charge_weights is None:
            self.charge_weights = np.random.normal(0, 0.1, 4)  # Xavier initialization

        # Simple linear model for model selection
        model_scores = {}
        for model in available_models:
            if model not in self.model_weights:
                # Xavier initialization for model weights
                self.model_weights[model] = np.random.normal(0, 0.1, 4)

            # Pure learned model selection - no hardcoded scoring
            model_scores[model] = np.dot(features, self.model_weights[model])

        selected_model = max(model_scores.keys(), key=lambda x: model_scores[x])

        # Learnable linear model for charging decision
        charge_score = np.dot(features, self.charge_weights)
        should_charge = charge_score > 0  # Use 0 as threshold (sigmoid-like)

        return selected_model, should_charge

    def compute_loss(
        self,
        prediction: Tuple[str, bool],
        target: Tuple[str, bool],
        features: np.ndarray,
        available_models: Dict[str, Dict[str, float]],
    ) -> float:
        """Compute loss function: α*(1-accuracy) + β*latency + γ*non_clean_energy"""
        pred_model, pred_charge = prediction
        target_model, target_charge = target

        # Model selection loss
        model_correct = 1.0 if pred_model == target_model else 0.0

        # Charging loss
        charge_correct = 1.0 if pred_charge == target_charge else 0.0

        # Combined accuracy
        total_accuracy = (model_correct + charge_correct) / 2.0

        # Latency penalty (mock based on model)
        latency_penalty = available_models[pred_model]["latency"] / 3000.0

        # Clean energy penalty
        clean_energy_penalty = 0.0
        if target_charge and not pred_charge:
            clean_energy_penalty = 1.0 - features[1]  # Missed clean energy opportunity

        loss = (
            self.weights["accuracy_weight"] * (1 - total_accuracy)
            + self.weights["latency_weight"] * latency_penalty
            + self.weights["clean_energy_weight"] * clean_energy_penalty
        )

        return loss

    def train_step(
        self,
        scenario: Dict,
        available_models: Dict[str, Dict[str, float]],
        learning_rate: float = 0.01,
    ) -> float:
        """Single training step using proper gradient descent."""
        features = self.extract_features(scenario)
        target_model = scenario["optimal_model"]
        target_charge = scenario["should_charge"]

        # Forward pass
        prediction = self.predict_model_and_charge(
            features, list(available_models.keys()), available_models
        )

        # Compute loss
        loss = self.compute_loss(
            prediction, (target_model, target_charge), features, available_models
        )

        # Backward pass - proper gradient computation
        pred_model, pred_charge = prediction

        # Model selection gradient (cross-entropy-like)
        if pred_model != target_model:
            # Increase weights for correct model, decrease for incorrect model
            if target_model in self.model_weights:
                self.model_weights[target_model] += learning_rate * features * 0.1
            if pred_model in self.model_weights:
                self.model_weights[pred_model] -= learning_rate * features * 0.1

        # Charge decision gradient (binary cross-entropy-like)
        target_charge_val = 1.0 if target_charge else 0.0
        pred_charge_val = 1.0 if pred_charge else 0.0

        # Gradient for charge weights
        if self.charge_weights is not None:
            charge_gradient = (pred_charge_val - target_charge_val) * features
            self.charge_weights -= learning_rate * charge_gradient * 0.1

        return loss

    def split_data(
        self, data: List[Dict], train_ratio: float = 0.7, val_ratio: float = 0.2
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split data into train/validation/test sets."""
        # Convert to indices for shuffling
        indices = list(range(len(data)))
        np.random.shuffle(indices)

        n = len(data)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]

        train_data = [data[i] for i in train_indices]
        val_data = [data[i] for i in val_indices]
        test_data = [data[i] for i in test_indices]

        print(
            f"Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}"
        )
        return train_data, val_data, test_data

    def evaluate(
        self, data: List[Dict], available_models: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Evaluate model on validation/test data."""
        total_loss = 0.0
        model_correct = 0
        charge_correct = 0

        for scenario in data:
            features = self.extract_features(scenario)
            target_model = scenario["optimal_model"]
            target_charge = scenario["should_charge"]

            # Forward pass
            prediction = self.predict_model_and_charge(
                features, list(available_models.keys()), available_models
            )

            # Compute loss
            loss = self.compute_loss(
                prediction, (target_model, target_charge), features, available_models
            )
            total_loss += loss

            # Track accuracy
            pred_model, pred_charge = prediction
            if pred_model == target_model:
                model_correct += 1
            if pred_charge == target_charge:
                charge_correct += 1

        n = len(data)
        return {
            "loss": total_loss / n,
            "model_accuracy": model_correct / n,
            "charge_accuracy": charge_correct / n,
            "overall_accuracy": (model_correct + charge_correct) / (2 * n),
        }

    def train(
        self,
        training_data: List[Dict],
        available_models: Dict[str, Dict[str, float]],
        epochs: int = 10000,
        learning_rate: float = 0.01,
    ):
        """Train CustomController with train/validation/test split."""
        print(f"Training CustomController for {epochs} epochs...")

        # Split data
        train_data, val_data, test_data = self.split_data(training_data)

        best_val_loss = float("inf")
        patience = 50
        patience_counter = 0
        best_weights = None
        final_epoch = 0

        for epoch in range(epochs):
            final_epoch = epoch
            # Training phase
            total_loss = 0.0
            # Shuffle training data using indices
            train_indices = list(range(len(train_data)))
            np.random.shuffle(train_indices)
            shuffled_train_data = [train_data[i] for i in train_indices]

            for scenario in shuffled_train_data:
                loss = self.train_step(scenario, available_models, learning_rate)
                total_loss += loss

            train_loss = total_loss / len(train_data)

            # Validation phase
            val_metrics = self.evaluate(val_data, available_models)

            # Early stopping
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0
                # Save best weights
                best_weights = {
                    "model_weights": {
                        k: v.copy() for k, v in self.model_weights.items()
                    },
                    "charge_weights": (
                        self.charge_weights.copy()
                        if self.charge_weights is not None
                        else None
                    ),
                }
            else:
                patience_counter += 1

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_metrics['loss']:.4f}"
                )
                print(
                    f"  Val Acc: Model={val_metrics['model_accuracy']:.3f}, Charge={val_metrics['charge_accuracy']:.3f}"
                )

            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # Restore best weights
        if best_weights:
            self.model_weights = best_weights["model_weights"]
            self.charge_weights = best_weights["charge_weights"]

        # Final evaluation on test set
        print("\n📊 Final Evaluation:")
        test_metrics = self.evaluate(test_data, available_models)
        print(f"Test Loss: {test_metrics['loss']:.4f}")
        print(f"Test Model Accuracy: {test_metrics['model_accuracy']:.3f}")
        print(f"Test Charge Accuracy: {test_metrics['charge_accuracy']:.3f}")
        print(f"Test Overall Accuracy: {test_metrics['overall_accuracy']:.3f}")

        # Add training metadata
        test_metrics["training_epochs"] = final_epoch + 1
        test_metrics["training_samples"] = len(training_data)

        return test_metrics

    def save_weights(
        self, filepath: str, evaluation_stats: Optional[Dict[str, float]] = None
    ):
        """Save trained weights to JSON file."""
        weights_data = {
            "weights": self.weights,
            "model_weights": {k: v.tolist() for k, v in self.model_weights.items()},
            "charge_weights": (
                self.charge_weights.tolist()
                if self.charge_weights is not None
                else None
            ),
            "charge_threshold": self.charge_threshold,
        }

        if evaluation_stats:
            weights_data["evaluation_stats"] = evaluation_stats

        with open(filepath, "w") as f:
            json.dump(weights_data, f, indent=2)

        print(f"Trained weights saved to {filepath}")

    def load_weights(self, filepath: str):
        """Load trained weights from JSON file."""
        with open(filepath, "r") as f:
            weights_data = json.load(f)

        self.weights = weights_data["weights"]
        self.model_weights = {
            k: np.array(v) for k, v in weights_data["model_weights"].items()
        }
        self.charge_threshold = weights_data["charge_threshold"]

        print(f"Trained weights loaded from {filepath}")


def load_power_profiles() -> Dict[str, Dict[str, float]]:
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
        # Use real accuracy and latency from model-data.csv
        real_data = model_data.get(model_name, {})
        models[model_name] = {
            "accuracy": real_data.get("accuracy", 85.0),  # Fallback to 85% if not found
            "latency": real_data.get(
                "latency", data["avg_inference_time_seconds"] * 1000
            ),  # Use real latency, fallback to power profile
            "power_cost": data[
                "model_power_mw"
            ],  # Keep power data from power profiling
        }

    return models


def main():
    """Train CustomController from scratch and save weights."""
    print("Loading training data...")
    try:
        training_data = CustomController().load_training_data(
            "results/training_data.json"
        )
        print(f"Loaded {len(training_data)} training samples")
    except FileNotFoundError:
        print("Training data not found. Please run generate_training_data.py first.")
        return

    print("Loading power profiles...")
    available_models = load_power_profiles()

    print("Initializing CustomController...")
    controller = CustomController()

    print("Starting training...")
    evaluation_stats = controller.train(
        training_data, available_models, epochs=10000, learning_rate=0.01
    )

    print("Saving trained weights...")
    controller.save_weights("results/custom_controller_weights.json", evaluation_stats)

    print("Training complete!")


if __name__ == "__main__":
    main()
