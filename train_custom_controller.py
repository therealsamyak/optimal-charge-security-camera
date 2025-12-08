#!/usr/bin/env python3
"""
CustomController training using PyTorch neural networks.
Trains model selection and charging decisions using MIPS-generated training data.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from src.neural_controller import NeuralController, NeuralLoss


class CustomController:
    """Custom controller with neural network for model selection and charging."""

    def __init__(self):
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.model = NeuralController().to(self.device)
        self.criterion = NeuralLoss()
        self.optimizer = None
        self.scheduler = None
        self.model_to_idx = {}
        self.idx_to_model = {}
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

    def extract_features(self, scenario: Dict) -> torch.Tensor:
        """Extract features from training scenario."""
        features = np.array(
            [
                scenario["battery_level"] / 100.0,
                scenario["clean_energy_percentage"] / 100.0,
                scenario["accuracy_requirement"],  # Already 0-1 range
                scenario["latency_requirement"] / 30.0,  # Normalize to 30ms max
            ]
        )
        return torch.FloatTensor(features).to(self.device)

    def predict_model_and_charge(
        self,
        features: torch.Tensor,
        available_models: List[str],
        model_data: Dict[str, Dict[str, float]],
    ) -> Tuple[str, bool]:
        """Predict model selection and charging decision using neural network."""
        self.model.eval()
        with torch.no_grad():
            model_probs, charge_prob = self.model(features.unsqueeze(0))

        # Get model prediction
        model_idx = int(torch.argmax(model_probs, dim=-1).item())
        selected_model = self.idx_to_model[model_idx]

        # Get charge decision
        should_charge = charge_prob.item() > 0.5

        return selected_model, should_charge

    def compute_loss(
        self,
        model_probs: torch.Tensor,
        charge_prob: torch.Tensor,
        target_model_idx: torch.Tensor,
        target_charge: torch.Tensor,
    ) -> torch.Tensor:
        """Compute neural network loss."""
        return self.criterion(model_probs, charge_prob, target_model_idx, target_charge)

    def setup_model_mapping(self, available_models: List[str]):
        """Setup model to index mapping for neural network."""
        self.model_to_idx = {
            model: int(idx) for idx, model in enumerate(available_models)
        }
        self.idx_to_model = {
            int(idx): model for idx, model in enumerate(available_models)
        }

    def train_step(
        self,
        scenario: Dict,
    ) -> float:
        """Single training step using PyTorch."""
        features = self.extract_features(scenario)
        target_model_idx = torch.tensor(
            [self.model_to_idx[scenario["optimal_model"]]], dtype=torch.long
        ).to(self.device)
        target_charge = torch.tensor([float(scenario["should_charge"])]).to(self.device)

        # Forward pass
        self.model.train()
        model_probs, charge_prob = self.model(features.unsqueeze(0))

        # Compute loss
        loss = self.compute_loss(
            model_probs, charge_prob, target_model_idx, target_charge
        )

        # Backward pass
        if self.optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, patience=10
            )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

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
        self, data: List[Dict], available_models: List[str]
    ) -> Dict[str, float]:
        """Evaluate model on validation/test data."""
        total_loss = 0.0
        model_correct = 0
        charge_correct = 0

        self.model.eval()
        with torch.no_grad():
            for scenario in data:
                features = self.extract_features(scenario)
                target_model_idx = torch.tensor(
                    [self.model_to_idx[scenario["optimal_model"]]], dtype=torch.long
                ).to(self.device)
                target_charge = torch.tensor([float(scenario["should_charge"])]).to(
                    self.device
                )

                # Forward pass
                model_probs, charge_prob = self.model(features.unsqueeze(0))

                # Compute loss
                loss = self.compute_loss(
                    model_probs, charge_prob, target_model_idx, target_charge
                )
                total_loss += loss.item()

                # Track accuracy
                pred_model_idx = int(torch.argmax(model_probs, dim=-1).item())
                pred_model = self.idx_to_model[pred_model_idx]
                pred_charge = charge_prob.item() > 0.5

                if pred_model == scenario["optimal_model"]:
                    model_correct += 1
                if pred_charge == scenario["should_charge"]:
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
        epochs: int = 1000,
        learning_rate: float = 0.001,
    ):
        """Train CustomController with train/validation/test split."""
        print(f"Training CustomController for {epochs} epochs...")

        # Setup model mapping
        model_list = list(available_models.keys())
        self.setup_model_mapping(model_list)

        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=10
        )

        # Split data
        print("Splitting data into train/validation/test sets...")
        train_data, val_data, test_data = self.split_data(training_data)
        print(
            f"Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}"
        )

        best_val_loss = float("inf")
        patience = 50
        patience_counter = 0
        best_state_dict = None
        final_epoch = 0

        print("Starting epoch training loop...")
        for epoch in range(epochs):
            final_epoch = epoch

            # Progress logging every 100 epochs
            if (epoch + 1) % 100 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Training...")

            # Training phase
            total_loss = 0.0
            # Shuffle training data using indices
            train_indices = list(range(len(train_data)))
            np.random.shuffle(train_indices)
            shuffled_train_data = [train_data[i] for i in train_indices]

            for scenario in shuffled_train_data:
                loss = self.train_step(scenario)
                total_loss += loss

            train_loss = total_loss / len(train_data)

            # Validation phase
            val_metrics = self.evaluate(val_data, model_list)

            # Learning rate scheduling
            self.scheduler.step(val_metrics["loss"])

            # Early stopping
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0
                best_state_dict = self.model.state_dict().copy()
            else:
                patience_counter += 1

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_metrics['loss']:.4f}"
                )
                print(
                    f"  Val Acc: Model={val_metrics['model_accuracy']:.3f}, Charge={val_metrics['charge_accuracy']:.3f}"
                )
                print(f"  Patience counter: {patience_counter}/50")

            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # Restore best weights
        if best_state_dict:
            self.model.load_state_dict(best_state_dict)

        # Final evaluation on test set
        print("\n📊 Final Evaluation:")
        test_metrics = self.evaluate(test_data, model_list)
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
        """Save trained neural network to JSON file."""
        weights_data = {
            "model_state_dict": {
                k: v.tolist() for k, v in self.model.state_dict().items()
            },
            "model_to_idx": self.model_to_idx,
            "idx_to_model": self.idx_to_model,
        }

        if evaluation_stats:
            weights_data["evaluation_stats"] = evaluation_stats

        with open(filepath, "w") as f:
            json.dump(weights_data, f, indent=2)

        print(f"Trained neural network saved to {filepath}")

    def load_weights(self, filepath: str):
        """Load trained neural network from JSON file."""
        with open(filepath, "r") as f:
            weights_data = json.load(f)

        # Load state dict
        state_dict = {
            k: torch.tensor(v) for k, v in weights_data["model_state_dict"].items()
        }
        self.model.load_state_dict(state_dict)

        # Load mappings
        self.model_to_idx = weights_data["model_to_idx"]
        self.idx_to_model = {int(k): v for k, v in weights_data["idx_to_model"].items()}

        print(f"Trained neural network loaded from {filepath}")


def load_power_profiles() -> Dict[str, Dict[str, float]]:
    """Load power profiles using PowerProfiler."""
    from pathlib import Path
    import sys

    sys.path.insert(0, str(Path(__file__).parent / "src"))

    from src.power_profiler import PowerProfiler

    profiler = PowerProfiler()
    profiler.load_profiles()  # Load profiles from file
    return profiler.get_all_models_data()


def main():
    """Train CustomController from scratch and save weights."""
    print("Loading training data...")
    try:
        training_data = CustomController().load_training_data(
            "results/training_data.json"
        )
        print(f"✓ Loaded {len(training_data)} training samples")
    except FileNotFoundError:
        print("✗ Training data not found. Please run generate_training_data.py first.")
        return
    except Exception as e:
        print(f"✗ Error loading training data: {e}")
        return

    print("Loading power profiles...")
    try:
        available_models = load_power_profiles()
        print(f"✓ Loaded {len(available_models)} model profiles")
        if not available_models:
            print("✗ No model profiles found")
            return
    except Exception as e:
        print(f"✗ Error loading power profiles: {e}")
        return

    print("Initializing CustomController...")
    try:
        controller = CustomController()
        print("✓ CustomController initialized")
    except Exception as e:
        print(f"✗ Error initializing controller: {e}")
        return

    print("Starting training...")
    try:
        print("Training parameters: epochs=10000, learning_rate=0.01")
        print(f"Training data size: {len(training_data)} samples")
        print(f"Available models: {list(available_models.keys())}")
        print("Beginning training epochs...")

        evaluation_stats = controller.train(
            training_data, available_models, epochs=10000, learning_rate=0.01
        )
        print("✓ Training complete!")
    except Exception as e:
        print(f"✗ Error during training: {e}")
        return

    print("Saving trained weights...")
    try:
        controller.save_weights(
            "results/custom_controller_weights.json", evaluation_stats
        )
        print("✓ Controller weights saved to results/custom_controller_weights.json")
    except Exception as e:
        print(f"✗ Error saving weights: {e}")
        return

    print("✓ Training pipeline completed successfully!")


if __name__ == "__main__":
    main()
