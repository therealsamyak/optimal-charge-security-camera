#!/usr/bin/env python3
"""
Import and use trained controller from PyTorch format.
"""

import torch

from .neural_controller import NeuralController


def import_controller(export_path: str = "controller-unified.pt"):
    """Import controller from PyTorch format."""
    print(f"📥 Importing controller from {export_path}...")

    # Load export data
    export_data = torch.load(export_path, map_location="cpu")

    # Create model and load state
    model = NeuralController()
    model.load_state_dict(export_data["model_state_dict"])
    model.eval()

    print(f"✅ Loaded {export_data['input_features']}-input controller")
    print(f"🎯 Output classes: {export_data['output_classes']}")

    return model, export_data


def predict_batch(model, export_data, input_batch):
    """Make batch predictions."""
    x = torch.tensor(input_batch, dtype=torch.float32)

    with torch.no_grad():
        model_probs, charge_probs = model(x)

    # Get predictions
    model_indices = torch.argmax(model_probs, dim=1)
    charge_decisions = charge_probs > 0.5

    # Convert to model names
    idx_to_model = export_data["model_mappings"]["idx_to_model"]
    selected_models = [
        idx_to_model.get(str(idx.item()), f"Unknown_{idx.item()}")
        for idx in model_indices
    ]

    return selected_models, charge_decisions.tolist(), model_probs.tolist()


def main():
    """Test imported controller."""
    # Import controller
    model, export_data = import_controller()

    # Test with multiple inputs
    test_inputs = [
        [0.5, 0.8, 0.75, 0.5, 0.33, 0.9, 0.05],  # Medium battery, high clean energy
        [0.1, 0.2, 0.75, 0.5, 0.33, 0.9, 0.05],  # Low battery, low clean energy
        [0.9, 0.95, 0.75, 0.5, 0.33, 0.7, 0.08],  # High battery, very high clean energy
    ]

    print(f"\n🎯 Testing with {len(test_inputs)} sample inputs...")

    # Make predictions
    selected_models, charge_decisions, model_probs = predict_batch(
        model, export_data, test_inputs
    )

    # Display results
    for i, (input_vec, model_name, should_charge, probs) in enumerate(
        zip(test_inputs, selected_models, charge_decisions, model_probs)
    ):
        print(f"\n--- Sample {i + 1} ---")
        print(f"📊 Input: {input_vec}")
        print(f"🤖 Selected: {model_name}")
        print(f"🔋 Charge: {should_charge}")
        print("📈 Top 3 Models:")

        # Sort by probability
        sorted_probs = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:3]
        for idx, prob in sorted_probs:
            model_name = export_data["model_mappings"]["idx_to_model"].get(
                str(idx), f"Model_{idx}"
            )
            print(f"   {model_name}: {prob:.3f}")


if __name__ == "__main__":
    main()
