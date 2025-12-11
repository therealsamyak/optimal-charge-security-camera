#!/usr/bin/env python3
"""
Export trained controller to PyTorch format for easy importing.
"""

import torch
import json

from .neural_controller import NeuralController


def export_controller(
    controller_path: str = "controller-unified.json",
    export_path: str = "controller-unified.pt",
):
    """Export controller to PyTorch format."""
    print(f"📦 Exporting controller from {controller_path} to {export_path}...")

    # Load JSON data
    with open(controller_path, "r") as f:
        controller_data = json.load(f)

    # Create model and load weights
    model = NeuralController()
    state_dict = controller_data["model_state_dict"]

    # Convert lists back to tensors
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

    # Create export package
    export_data = {
        "model_state_dict": model.state_dict(),
        "model_mappings": controller_data["model_mappings"],
        "metadata": controller_data["training_info"],
        "input_features": controller_data["input_features"],
        "output_classes": controller_data["output_classes"],
    }

    # Save PyTorch format
    torch.save(export_data, export_path)
    print(f"✅ Controller exported to {export_path}")

    # Also save as script module for deployment
    script_path = export_path.replace(".pt", "_scripted.pt")
    scripted_model = torch.jit.script(model)
    scripted_model.save(script_path)
    print(f"✅ Scripted model saved to {script_path}")

    return export_path, script_path


def main():
    """Export the trained controller."""
    export_path, script_path = export_controller()

    print("\n📋 Export Summary:")
    print(f"🔧 PyTorch Model: {export_path}")
    print(f"🚀 Scripted Model: {script_path}")
    print("\n💡 Usage:")
    print("   # Load PyTorch format:")
    print(f"   export_data = torch.load('{export_path}')")
    print("   model = NeuralController()")
    print("   model.load_state_dict(export_data['model_state_dict'])")
    print("   ")
    print("   # Or load scripted model:")
    print(f"   model = torch.jit.load('{script_path}')")


if __name__ == "__main__":
    main()
