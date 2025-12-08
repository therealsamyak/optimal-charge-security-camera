#!/usr/bin/env python3
"""
CustomController training using PyTorch neural networks.
Trains model selection and charging decisions using MIPS-generated training data.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from src.neural_controller import NeuralController, NeuralLoss
from src.logging_config import setup_logging, get_logger

# Initialize logging
logger = setup_logging()


class CustomController:
    """Custom controller with neural network for model selection and charging."""

    def __init__(self):
        logger.info("Initializing CustomController")

        # Device selection with detailed logging
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("Using MPS (Apple Silicon) device")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info(f"Using CUDA device - {torch.cuda.get_device_name()}")
            logger.info(
                f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
            )
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU device")

        logger.debug(f"Selected device: {self.device}")

        # Initialize neural network
        self.model = NeuralController().to(self.device)
        logger.info("NeuralController model initialized")
        logger.debug(f"Model architecture: {self.model}")

        # Initialize loss function
        self.criterion = NeuralLoss()
        logger.info("NeuralLoss criterion initialized")

        # Training components
        self.optimizer = None
        self.scheduler = None
        self.model_to_idx = {}
        self.idx_to_model = {}
        self.logger = get_logger(self.__class__.__name__)

        logger.info("CustomController initialization complete")

    def load_training_data(self, filepath: str) -> List[Dict]:
        """Load training data from JSON file."""
        logger.info(f"Loading training data from {filepath}")

        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            logger.info(f"✓ Successfully loaded {len(data)} training samples")

            # Log data statistics
            if data:
                sample = data[0]
                logger.debug(f"Sample training data keys: {list(sample.keys())}")
                logger.debug(
                    f"Sample battery level: {sample.get('battery_level', 'N/A')}"
                )
                logger.debug(
                    f"Sample clean energy: {sample.get('clean_energy_percentage', 'N/A')}%"
                )
                logger.debug(
                    f"Sample optimal model: {sample.get('optimal_model', 'N/A')}"
                )
                logger.debug(
                    f"Sample should_charge: {sample.get('should_charge', 'N/A')}"
                )

                # Count unique models
                models = set(
                    s.get("optimal_model") for s in data if "optimal_model" in s
                )
                logger.info(f"Unique models in training data: {len(models)}")
                logger.debug(f"Models: {sorted(models)}")

                # Count locations
                locations = set(s.get("location") for s in data if "location" in s)
                logger.info(f"Unique locations in training data: {len(locations)}")
                logger.debug(f"Locations: {sorted(locations)}")

            return data

        except FileNotFoundError:
            logger.error(f"Training data file not found: {filepath}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in training data file: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            logger.exception("Full traceback for training data loading error")
            raise

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
        try:
            features = np.array(
                [
                    scenario["battery_level"] / 100.0,
                    scenario["clean_energy_percentage"] / 100.0,
                    scenario["accuracy_requirement"],  # Already 0-1 range
                    scenario["latency_requirement"] / 30.0,  # Normalize to 30ms max
                ]
            )

            # Log feature extraction details for debugging
            logger.debug(f"Extracted features for scenario:")
            logger.debug(
                f"  Battery: {scenario['battery_level']:.1f}% -> {features[0]:.3f}"
            )
            logger.debug(
                f"  Clean Energy: {scenario['clean_energy_percentage']:.1f}% -> {features[1]:.3f}"
            )
            logger.debug(
                f"  Accuracy Req: {scenario['accuracy_requirement']:.3f} -> {features[2]:.3f}"
            )
            logger.debug(
                f"  Latency Req: {scenario['latency_requirement']}ms -> {features[3]:.3f}"
            )

            tensor = torch.FloatTensor(features).to(self.device)
            logger.debug(
                f"Feature tensor shape: {tensor.shape}, device: {tensor.device}"
            )

            return tensor

        except KeyError as e:
            logger.error(f"Missing required key in scenario: {e}")
            logger.debug(f"Available keys: {list(scenario.keys())}")
            raise
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            raise

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
        try:
            # Extract features and targets
            features = self.extract_features(scenario)
            target_model_idx = torch.tensor(
                [self.model_to_idx[scenario["optimal_model"]]], dtype=torch.long
            ).to(self.device)
            target_charge = torch.tensor([float(scenario["should_charge"])]).to(
                self.device
            )

            logger.debug(
                f"Training step - Target model: {scenario['optimal_model']} (idx: {target_model_idx.item()})"
            )
            logger.debug(
                f"Training step - Target charge: {scenario['should_charge']} (tensor: {target_charge.item():.1f})"
            )

            # Initialize optimizer if needed
            if self.optimizer is None:
                self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, patience=10
                )
                logger.info("Optimizer and scheduler initialized")
                logger.debug(f"Optimizer: {self.optimizer}")
                logger.debug(f"Scheduler: {self.scheduler}")

            # Forward pass
            self.model.train()
            model_probs, charge_prob = self.model(features.unsqueeze(0))

            logger.debug(f"Model output - Model probs shape: {model_probs.shape}")
            logger.debug(f"Model output - Charge prob: {charge_prob.item():.4f}")

            # Compute loss
            loss = self.compute_loss(
                model_probs, charge_prob, target_model_idx, target_charge
            )
            loss_value = loss.item()

            logger.debug(f"Computed loss: {loss_value:.6f}")

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Log gradient information
            total_norm = 0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1.0 / 2)
            logger.debug(f"Gradient norm: {total_norm:.6f}")

            self.optimizer.step()

            # Log learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            logger.debug(f"Current learning rate: {current_lr:.6f}")

            return loss_value

        except Exception as e:
            logger.error(f"Error in training step: {e}")
            logger.exception("Full traceback for training step error")
            raise

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
        logger.info("=" * 80)
        logger.info("STARTING CUSTOM CONTROLLER TRAINING")
        logger.info("=" * 80)
        logger.info(
            f"Training parameters: epochs={epochs}, learning_rate={learning_rate}"
        )
        logger.info(f"Training data size: {len(training_data)} samples")
        logger.info(f"Available models: {list(available_models.keys())}")

        print(f"Training CustomController for {epochs} epochs...")

        # Setup model mapping
        model_list = list(available_models.keys())
        self.setup_model_mapping(model_list)
        logger.info(f"Model mapping setup: {len(model_list)} models")
        logger.debug(f"Model to index mapping: {self.model_to_idx}")

        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=10
        )
        logger.info("Optimizer and scheduler initialized")
        logger.debug(f"Initial learning rate: {learning_rate}")

        # Split data
        print("Splitting data into train/validation/test sets...")
        logger.info("Splitting data into train/validation/test sets...")
        train_data, val_data, test_data = self.split_data(training_data)
        print(
            f"Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}"
        )
        logger.info(
            f"Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}"
        )

        best_val_loss = float("inf")
        patience = 50
        patience_counter = 0
        best_state_dict = None
        final_epoch = 0
        training_start_time = datetime.now()

        print("Starting epoch training loop...")
        logger.info("Starting epoch training loop...")
        for epoch in range(epochs):
            final_epoch = epoch
            epoch_start_time = datetime.now()

            # Progress logging every 100 epochs
            if (epoch + 1) % 100 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Training...")
                logger.info(f"Epoch {epoch + 1}/{epochs} - Training...")

            # Training phase
            total_loss = 0.0
            # Shuffle training data using indices
            train_indices = list(range(len(train_data)))
            np.random.shuffle(train_indices)
            shuffled_train_data = [train_data[i] for i in train_indices]

            for i, scenario in enumerate(shuffled_train_data):
                loss = self.train_step(scenario)
                total_loss += loss

                # Log progress for large datasets
                if len(train_data) > 10000 and i % 5000 == 0:
                    logger.debug(
                        f"Epoch {epoch}: Processed {i}/{len(train_data)} training samples"
                    )

            train_loss = total_loss / len(train_data)
            logger.debug(f"Epoch {epoch}: Average training loss: {train_loss:.6f}")

            # Validation phase
            val_metrics = self.evaluate(val_data, model_list)
            logger.debug(f"Epoch {epoch}: Validation metrics: {val_metrics}")

            # Learning rate scheduling
            self.scheduler.step(val_metrics["loss"])
            current_lr = self.optimizer.param_groups[0]["lr"]
            logger.debug(
                f"Epoch {epoch}: Learning rate after scheduler: {current_lr:.6f}"
            )

            # Early stopping
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0
                best_state_dict = self.model.state_dict().copy()
                logger.debug(
                    f"Epoch {epoch}: New best validation loss: {best_val_loss:.6f}"
                )
            else:
                patience_counter += 1
                logger.debug(
                    f"Epoch {epoch}: Patience counter: {patience_counter}/{patience}"
                )

            if epoch % 10 == 0:
                epoch_time = (datetime.now() - epoch_start_time).total_seconds()
                print(
                    f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_metrics['loss']:.4f}"
                )
                print(
                    f"  Val Acc: Model={val_metrics['model_accuracy']:.3f}, Charge={val_metrics['charge_accuracy']:.3f}"
                )
                print(f"  Patience counter: {patience_counter}/50")
                logger.info(
                    f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_metrics['loss']:.4f}, Time={epoch_time:.2f}s"
                )
                logger.info(
                    f"  Val Acc: Model={val_metrics['model_accuracy']:.3f}, Charge={val_metrics['charge_accuracy']:.3f}"
                )
                logger.info(
                    f"  Patience counter: {patience_counter}/50, LR: {current_lr:.6f}"
                )

            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                logger.info(f"Early stopping at epoch {epoch} due to patience limit")
                break

        # Restore best weights
        if best_state_dict:
            self.model.load_state_dict(best_state_dict)
            logger.info(
                "Restored best model weights from epoch with lowest validation loss"
            )

        training_time = (datetime.now() - training_start_time).total_seconds()
        logger.info(
            f"Training completed in {training_time:.2f} seconds over {final_epoch + 1} epochs"
        )

        # Final evaluation on test set
        print("\n📊 Final Evaluation:")
        logger.info("Final evaluation on test set...")
        test_metrics = self.evaluate(test_data, model_list)
        print(f"Test Loss: {test_metrics['loss']:.4f}")
        print(f"Test Model Accuracy: {test_metrics['model_accuracy']:.3f}")
        print(f"Test Charge Accuracy: {test_metrics['charge_accuracy']:.3f}")
        print(f"Test Overall Accuracy: {test_metrics['overall_accuracy']:.3f}")
        logger.info(f"Test Loss: {test_metrics['loss']:.4f}")
        logger.info(f"Test Model Accuracy: {test_metrics['model_accuracy']:.3f}")
        logger.info(f"Test Charge Accuracy: {test_metrics['charge_accuracy']:.3f}")
        logger.info(f"Test Overall Accuracy: {test_metrics['overall_accuracy']:.3f}")

        # Add training metadata
        test_metrics["training_epochs"] = final_epoch + 1
        test_metrics["training_samples"] = len(training_data)
        test_metrics["training_time_seconds"] = training_time
        test_metrics["best_validation_loss"] = best_val_loss

        logger.info("=" * 80)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)

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
    logger.info("=" * 80)
    logger.info("STARTING CUSTOM CONTROLLER TRAINING PIPELINE")
    logger.info("=" * 80)

    start_time = datetime.now()
    logger.info(f"Pipeline start time: {start_time}")

    print("Loading training data...")
    logger.info("Loading training data...")
    try:
        controller = CustomController()
        training_data = controller.load_training_data("results/training_data.json")
        print(f"✓ Loaded {len(training_data)} training samples")
        logger.info(f"✓ Loaded {len(training_data)} training samples")
    except FileNotFoundError:
        print("✗ Training data not found. Please run generate_training_data.py first.")
        logger.error(
            "✗ Training data not found. Please run generate_training_data.py first."
        )
        return
    except Exception as e:
        print(f"✗ Error loading training data: {e}")
        logger.error(f"✗ Error loading training data: {e}")
        logger.exception("Full traceback for training data loading error")
        return

    print("Loading power profiles...")
    logger.info("Loading power profiles...")
    try:
        available_models = load_power_profiles()
        print(f"✓ Loaded {len(available_models)} model profiles")
        logger.info(f"✓ Loaded {len(available_models)} model profiles")
        if not available_models:
            print("✗ No model profiles found")
            logger.error("✗ No model profiles found")
            return
    except Exception as e:
        print(f"✗ Error loading power profiles: {e}")
        logger.error(f"✗ Error loading power profiles: {e}")
        logger.exception("Full traceback for power profiles loading error")
        return

    print("Initializing CustomController...")
    logger.info("Initializing CustomController...")
    try:
        controller = CustomController()
        print("✓ CustomController initialized")
        logger.info("✓ CustomController initialized")
    except Exception as e:
        print(f"✗ Error initializing controller: {e}")
        logger.error(f"✗ Error initializing controller: {e}")
        logger.exception("Full traceback for controller initialization error")
        return

    print("Starting training...")
    logger.info("Starting training...")
    try:
        print("Training parameters: epochs=10000, learning_rate=0.01")
        print(f"Training data size: {len(training_data)} samples")
        print(f"Available models: {list(available_models.keys())}")
        print("Beginning training epochs...")

        logger.info("Training parameters: epochs=10000, learning_rate=0.01")
        logger.info(f"Training data size: {len(training_data)} samples")
        logger.info(f"Available models: {list(available_models.keys())}")
        logger.info("Beginning training epochs...")

        evaluation_stats = controller.train(
            training_data, available_models, epochs=10000, learning_rate=0.01
        )
        print("✓ Training complete!")
        logger.info("✓ Training complete!")
        logger.info(f"Final evaluation stats: {evaluation_stats}")
    except Exception as e:
        print(f"✗ Error during training: {e}")
        logger.error(f"✗ Error during training: {e}")
        logger.exception("Full traceback for training error")
        return

    print("Saving trained weights...")
    logger.info("Saving trained weights...")
    try:
        controller.save_weights(
            "results/custom_controller_weights.json", evaluation_stats
        )
        print("✓ Controller weights saved to results/custom_controller_weights.json")
        logger.info(
            "✓ Controller weights saved to results/custom_controller_weights.json"
        )

        # Log file size
        weights_file = Path("results/custom_controller_weights.json")
        if weights_file.exists():
            file_size = weights_file.stat().st_size
            file_size_kb = file_size / 1024
            logger.info(f"Controller weights file size: {file_size_kb:.2f} KB")

    except Exception as e:
        print(f"✗ Error saving weights: {e}")
        logger.error(f"✗ Error saving weights: {e}")
        logger.exception("Full traceback for saving weights error")
        return

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print("✓ Training pipeline completed successfully!")
    logger.info(
        f"✓ Training pipeline completed successfully in {duration:.2f} seconds!"
    )
    logger.info("=" * 80)
    logger.info("CUSTOM CONTROLLER TRAINING PIPELINE COMPLETED")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
