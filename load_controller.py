#!/usr/bin/env python3
"""
Load and use the trained unified controller.
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.neural_controller import NeuralController


def load_controller(controller_path: str = "controller-unified.json"):
    """Load trained controller from JSON file."""
    with open(controller_path, "r") as f:
        controller_data = json.load(f)

    # Create model instance
    model = NeuralController()

    # Load weights
    state_dict = controller_data["model_state_dict"]
    model.shared_layers[0].weight.data = torch.tensor(
        state_dict["shared_layers.0.weight"]
    )
    model.shared_layers[0].bias.data = torch.tensor(state_dict["shared_layers.0.bias"])
    model.shared_layers[2].weight.data = torch.tensor(
        state_dict["shared_layers.2.weight"]
    )
    model.shared_layers[2].bias.data = torch.tensor(state_dict["shared_layers.2.bias"])
    model.model_head.weight.data = torch.tensor(state_dict["model_head.weight"])
    model.model_head.bias.data = torch.tensor(state_dict["model_head.bias"])
    model.charge_head.weight.data = torch.tensor(state_dict["charge_head.weight"])
    model.charge_head.bias.data = torch.tensor(state_dict["charge_head.bias"])

    # Set to evaluation mode
    model.eval()

    return model, controller_data


def predict_with_controller(model, controller_data, input_features):
    """Make prediction with loaded controller."""
    # Apply same normalization as training
    normalized_features = [
        input_features[0] / input_features[2],  # battery_level / battery_capacity_wh
        input_features[1] / 100.0,  # clean_energy_percentage / 100.0
        input_features[2] / 4.0,  # battery_capacity_wh / 4.0
        input_features[3] / 4.45,  # charge_rate_hours / 4.45
        input_features[4] / 600.0,  # task_interval_seconds / 600.0
        input_features[5] / 100.0,  # user_accuracy_requirement / 100.0
        input_features[6] / 0.5,  # user_latency_requirement / 0.5
    ]

    # Convert to tensor
    x = torch.tensor(normalized_features, dtype=torch.float32).unsqueeze(0)

    # Forward pass
    with torch.no_grad():
        model_probs, charge_prob = model(x)

    # Get predictions
    model_idx = torch.argmax(model_probs, dim=1).item()
    model_mappings = controller_data.get("model_mappings", {})
    idx_to_model = model_mappings.get("idx_to_model", {})
    selected_model = idx_to_model.get(str(model_idx), f"Unknown_{model_idx}")
    should_charge = charge_prob.item() > 0.5

    return selected_model, should_charge, model_probs.tolist()[0], charge_prob.item()


def main():
    """Test the loaded controller."""
    print("🔄 Loading trained controller...")

    # Load controller
    model, controller_data = load_controller()
    print(
        f"✅ Loaded {controller_data['model_type']} with {controller_data['input_features']} inputs"
    )

    # Test with sample input (raw values matching training data format)
    sample_input = [
        0.030275,  # battery_level (raw Wh)
        21.29,  # clean_energy_percentage
        0.06055,  # battery_capacity_wh
        1.0,  # charge_rate_hours
        300,  # task_interval_seconds
        90.5,  # user_accuracy_requirement
        0.01,  # user_latency_requirement
    ]

    print(f"\n🎯 Testing with sample input: {sample_input}")

    # Make prediction
    selected_model, should_charge, model_probs, charge_prob = predict_with_controller(
        model, controller_data, sample_input
    )

    print(f"🤖 Selected Model: {selected_model}")
    print(f"🔋 Should Charge: {should_charge} (confidence: {charge_prob:.3f})")
    print(f"📊 Model Probabilities:")
    model_mappings = controller_data.get("model_mappings", {})
    idx_to_model = model_mappings.get("idx_to_model", {})
    for i, prob in enumerate(model_probs):
        model_name = idx_to_model.get(str(i), f"Model_{i}")
        print(f"   {model_name}: {prob:.3f}")


if __name__ == "__main__":
    main()
