#!/usr/bin/env python3
"""
Test Unified Controller Training
Simple test to verify the unified controller works with production input format.
"""

import sys
import logging
import torch
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.neural_controller import NeuralController

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def test_unified_controller():
    """Test unified controller with production input format."""
    print("Testing unified controller with production input format...")

    # Test device detection (Apple Silicon)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS device (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA device")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")

    try:
        # Initialize controller
        model = NeuralController().to(device)

        logger.info(f"Model initialized on device: {device}")

        # Test with production input format
        # This matches exactly what you specified:
        production_input = {
            "battery_level": 2.8562001382594477,
            "clean_energy_percentage": 23.16,
            "battery_capacity_wh": 4,
            "charge_rate_hours": 4.45,
            "task_interval_seconds": 300,
            "user_accuracy_requirement": 90.5,
            "user_latency_requirement": 0.01,
        }

        # Extract features using production format
        features = torch.tensor(
            [
                production_input["battery_level"] / 100.0,  # Normalize to 0-1
                production_input["clean_energy_percentage"] / 100.0,  # Normalize to 0-1
                production_input["battery_capacity_wh"]
                / 4.0,  # Normalize by max capacity
                production_input["charge_rate_hours"] / 4.45,  # Normalize by max rate
                production_input["task_interval_seconds"]
                / 300.0,  # Normalize by max interval
                production_input["user_accuracy_requirement"]
                / 100.0,  # Normalize percentage to 0-1
                production_input["user_latency_requirement"]
                / 0.1,  # Normalize by 100ms max
            ],
            dtype=torch.float32,
        ).to(device)

        logger.debug(f"Input features shape: {features.shape}")

        # Forward pass
        model.eval()
        with torch.no_grad():
            model_probs, charge_prob = model(features.unsqueeze(0))

        logger.debug(
            f"Model outputs - model_probs: {model_probs.shape}, charge_prob: {charge_prob.shape}"
        )

        # Validate outputs
        assert model_probs.shape == (1, 7), f"Expected (1, 7), got {model_probs.shape}"
        assert charge_prob.shape == (1,), f"Expected (1,), got {charge_prob.shape}"

        # Check probability ranges
        assert torch.all(model_probs >= 0) and torch.all(model_probs <= 1), (
            "Model probs should be in [0,1]"
        )
        assert torch.all(charge_prob >= 0) and torch.all(charge_prob <= 1), (
            "Charge probs should be in [0,1]"
        )

        # Check sum to 1
        assert torch.allclose(
            model_probs.sum(dim=-1), torch.ones(1).to(device), atol=1e-6
        ), "Model probs should sum to 1"

        logger.info("✓ Unified controller test passed!")
        return True

    except Exception as e:
        logger.error(f"Unified controller test failed: {e}")
        return False


def main():
    """Run the unified controller test."""
    logger.info("🚀 Running Unified Controller Test")

    success = test_unified_controller()

    if success:
        logger.info(
            "✅ All tests passed! Unified controller is ready for implementation."
        )
        return 0
    else:
        logger.error("❌ Tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
