#!/usr/bin/env python3
"""
Train Unified Custom Controller
Combines all beam search training data into ONE unified controller.
Uses 7-input neural network with Apple Silicon MPS optimization.
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

from .neural_controller import NeuralController, NeuralLoss
from .power_profiler import load_power_profiles
import torch


def load_all_training_data(training_data_dir: str) -> list:
    """Load and combine all training data files with equal weighting."""
    print(f"📁 Loading training data from {training_data_dir}/...")

    # Find all training data files
    training_files = list(Path(training_data_dir).glob("*-training-data.json"))

    if not training_files:
        print("❌ No training data files found!")
        return []

    print(f"📊 Found {len(training_files)} training files:")
    for file in sorted(training_files):
        print(f"   - {file.name}")

    # Load all data
    all_training_data = []
    bucket_weights = {
        "success": 1.0,
        "most_clean_energy": 1.0,
        "least_total_energy": 1.0,
        "success_small_miss": 1.0,
    }

    for file_path in training_files:
        print(f"📖 Loading {file_path.name}...")
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            # Extract training samples
            training_samples = data.get("training_samples", [])
            if not training_samples:
                print(f"⚠️  No training samples found in {file_path.name}")
                continue

            # Apply equal weighting for each bucket
            bucket_name = file_path.stem.replace("-training-data", "")
            weight = bucket_weights.get(bucket_name, 1.0)

            # Add weight to each sample
            for sample in training_samples:
                sample["weight"] = weight

            all_training_data.extend(training_samples)

        except Exception as e:
            print(f"❌ Error loading {file_path.name}: {e}")
            continue

    print(f"📊 Combined dataset: {len(all_training_data)} total samples")

    # Calculate bucket distribution
    bucket_counts = {}
    for sample in all_training_data:
        bucket_name = sample.get("source_file", "").replace("-training-data.json", "")
        bucket_counts[bucket_name] = bucket_counts.get(bucket_name, 0) + 1

    print("📊 Bucket distribution:")
    for bucket, count in bucket_counts.items():
        print(f"   {bucket}: {count} samples")

    return all_training_data


def create_unified_controller():
    """Create and train the unified neural controller."""
    print("🧠 Creating unified neural controller...")

    # Initialize controller with Apple Silicon MPS optimization
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍎 Using MPS device (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 Using CUDA device")
    else:
        device = torch.device("cpu")
        print("💻 Using CPU device")

    # Create neural network with 7 inputs for production format
    model = NeuralController().to(device)
    loss_fn = NeuralLoss()

    # Load power profiles for model mapping
    available_models = load_power_profiles()
    print(f"⚡ Loaded {len(available_models)} power profiles")

    return model, loss_fn, available_models, device


def train_unified_controller(
    model, loss_fn, available_models, device, training_data, learning_rate=0.0005
):
    """Train the unified controller on all combined data."""
    print(f"🧠 Training unified controller on {len(training_data)} samples...")

    # Create model mappings
    model_to_idx = {}
    idx_to_model = {}

    # Collect all unique models from training data
    all_models = set()
    for sample in training_data:
        model_name = sample.get("optimal_model", "")
        if model_name and model_name not in ["IDLE"]:
            all_models.add(model_name)

    # Create mappings
    for i, model_name in enumerate(sorted(all_models)):
        model_to_idx[model_name] = i
        idx_to_model[i] = model_name

    print(f"🎯 Found {len(all_models)} unique models: {sorted(all_models)}")

    # Prepare training data
    features = []
    model_targets = []
    charge_targets = []
    weights = []

    for sample in training_data:
        # Extract features using production format
        feature_vector = torch.tensor(
            [
                sample["battery_level"] / 100.0,  # Normalize to 0-1
                sample["clean_energy_percentage"] / 100.0,  # Normalize to 0-1
                sample["battery_capacity_wh"] / 4.0,  # Normalize by max capacity
                sample["charge_rate_hours"] / 4.45,  # Normalize by max rate
                sample["task_interval_seconds"] / 300.0,  # Normalize by max interval
                sample["user_accuracy_requirement"]
                / 100.0,  # Normalize percentage to 0-1
                sample["user_latency_requirement"] / 0.1,  # Normalize by 100ms max
            ],
            dtype=torch.float32,
        )

        # Get target indices
        optimal_model = sample.get("optimal_model", "")
        if optimal_model in model_to_idx:
            target_idx = model_to_idx[optimal_model]
        else:
            # Skip samples with unknown models
            continue

        should_charge = float(sample.get("should_charge", False))

        features.append(feature_vector)
        model_targets.append(target_idx)
        charge_targets.append(should_charge)
        weights.append(sample.get("weight", 1.0))

    # Convert to tensors
    features_tensor = torch.stack(features).to(device)
    model_targets_tensor = torch.tensor(model_targets, dtype=torch.long).to(device)
    charge_targets_tensor = torch.tensor(charge_targets, dtype=torch.float32).to(device)
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    print(f"📊 Training data shape: {features_tensor.shape}")
    print(f"🎯 Model targets shape: {model_targets_tensor.shape}")
    print(f"⚡ Charge targets shape: {charge_targets_tensor.shape}")

    # Setup optimizer with adjusted learning rate for larger dataset
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training parameters
    epochs = 100
    batch_size = 512  # Optimized for Apple Silicon

    model.train()

    print(
        f"🚀 Starting training: {epochs} epochs, batch_size={batch_size}, lr={learning_rate}"
    )

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0

        # Mini-batch training
        for i in range(0, len(features_tensor), batch_size):
            start_idx = i
            end_idx = min(i + batch_size, len(features_tensor))

            batch_features = features_tensor[start_idx:end_idx]
            batch_model_targets = model_targets_tensor[start_idx:end_idx]
            batch_charge_targets = charge_targets_tensor[start_idx:end_idx]
            batch_weights = weights_tensor[start_idx:end_idx]

            # Forward pass
            model_probs, charge_prob = model(batch_features)

            # Compute weighted loss
            loss = loss_fn(
                model_probs, charge_prob, batch_model_targets, batch_charge_targets
            )
            weighted_loss = (loss * batch_weights).mean()

            # Backward pass
            optimizer.zero_grad()
            weighted_loss.backward()
            optimizer.step()

            epoch_loss += weighted_loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches

        # Log progress every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"📈 Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.6f}")

    print("✅ Training completed!")
    return model, model_to_idx, idx_to_model


def save_controller(
    model, available_models, output_path: str, model_to_idx, idx_to_model
):
    """Save the trained unified controller."""
    print(f"💾 Saving unified controller to {output_path}...")

    # Create evaluation statistics
    model.eval()

    # Save model state and mappings
    controller_data = {
        "model_type": "NeuralController",
        "input_features": 7,
        "output_classes": len(available_models),
        "model_state_dict": {
            "shared_layers.0.weight": model.shared_layers[0].weight.tolist(),
            "shared_layers.0.bias": model.shared_layers[0].bias.tolist(),
            "shared_layers.2.weight": model.shared_layers[2].weight.tolist(),
            "shared_layers.2.bias": model.shared_layers[2].bias.tolist(),
            "model_head.weight": model.model_head.weight.tolist(),
            "model_head.bias": model.model_head.bias.tolist(),
            "charge_head.weight": model.charge_head.weight.tolist(),
            "charge_head.bias": model.charge_head.bias.tolist(),
        },
        "model_mappings": {
            "model_to_idx": {v: k for k, v in model_to_idx.items()},
            "idx_to_model": {k: v for k, v in idx_to_model.items()},
        },
        "training_info": {
            "algorithm": "Supervised Multi-Task Learning (Imitation Learning)",
            "network_architecture": "7-input Multilayer Perceptron with dual heads",
            "loss_function": "Combined Cross-Entropy (0.5) + Binary Cross-Entropy (0.5)",
            "optimization": "Adam with learning rate 0.0005",
            "epochs": 100,
            "batch_size": 512,
            "device": "mps",
            "apple_silicon_optimized": True,
        },
        "timestamp": datetime.now().isoformat(),
        "version": "1.0",
    }

    # Save to file
    with open(output_path, "w") as f:
        json.dump(controller_data, f, indent=2)

    print(f"✅ Unified controller saved to {output_path}")


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description="Train unified custom controller")
    parser.add_argument(
        "--data-dir", type=str, default="training-data", help="Training data directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="controller-unified.json",
        help="Output controller file",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=512, help="Batch size for training"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0005,
        help="Learning rate for Adam optimizer",
    )

    args = parser.parse_args()

    start_time = datetime.now()
    print("=" * 80)
    print("🚀 UNIFIED CUSTOM CONTROLLER TRAINING")
    print("=" * 80)
    print(f"⏰ Start time: {start_time}")

    # Load all training data
    training_data = load_all_training_data(args.data_dir)

    if not training_data:
        print("❌ No training data available!")
        return 1

    # Create and train unified controller
    model, loss_fn, available_models, device = create_unified_controller()

    # Train the model
    trained_model, model_to_idx, idx_to_model = train_unified_controller(
        model, loss_fn, available_models, device, training_data, args.learning_rate
    )

    # Save the trained controller
    output_dir = Path(args.output).parent
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / args.output

    save_controller(
        trained_model, available_models, output_path, model_to_idx, idx_to_model
    )

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print("=" * 80)
    print("📊 TRAINING SUMMARY")
    print("=" * 80)
    print("✅ Training completed successfully!")
    print(f"📁 Total samples processed: {len(training_data)}")
    print(f"🎯 Unique models learned: {len(available_models)}")
    print(f"⏱️ Total training time: {duration:.2f} seconds")
    print(f"💾 Output file: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
