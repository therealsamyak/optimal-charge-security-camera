#!/usr/bin/env python3
"""
Test Neural Network Training
Tests forward/backward pass with 5 training scenarios
"""

import sys
import logging

import torch
from src.neural_controller import NeuralController, NeuralLoss

# Setup logging for tests
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def test_neural_controller_initialization():
    """Test neural network initialization."""
    logger.info("Testing NeuralController initialization...")

    # Test device detection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.debug("Using MPS device")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.debug("Using CUDA device")
    else:
        device = torch.device("cpu")
        logger.debug("Using CPU device")

    try:
        model = NeuralController().to(device)
        loss_fn = NeuralLoss()
        logger.debug(f"Model initialized on device: {device}")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise

    assert isinstance(model, NeuralController), (
        "Model should be NeuralController instance"
    )
    assert isinstance(loss_fn, NeuralLoss), "Loss should be NeuralLoss instance"

    # Test architecture
    test_input = torch.randn(2, 4).to(device)
    logger.debug(f"Test input shape: {test_input.shape}")

    try:
        model_probs, charge_prob = model(test_input)
        logger.debug(
            f"Model output shapes: model_probs={model_probs.shape}, charge_prob={charge_prob.shape}"
        )
    except Exception as e:
        logger.error(f"Forward pass failed: {e}")
        raise

    assert model_probs.shape == (2, 6), f"Expected (2, 6), got {model_probs.shape}"
    assert charge_prob.shape == (2,), f"Expected (2,), got {charge_prob.shape}"

    # Test probability ranges
    assert torch.all(model_probs >= 0) and torch.all(model_probs <= 1), (
        "Model probs should be in [0,1]"
    )
    assert torch.all(charge_prob >= 0) and torch.all(charge_prob <= 1), (
        "Charge probs should be in [0,1]"
    )
    assert torch.allclose(
        model_probs.sum(dim=-1), torch.ones(2).to(device), atol=1e-6
    ), "Model probs should sum to 1"

    logger.info("✓ NeuralController initialization test passed")


def test_forward_pass():
    """Test forward pass with different inputs."""
    logger.info("Testing forward pass...")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.debug(f"Using device: {device}")

    try:
        model = NeuralController().to(device)
        logger.debug("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    # Test scenarios
    scenarios = [
        [
            0.2,
            0.8,
            0.9,
            0.5,
        ],  # Low battery, high clean energy, high accuracy, medium latency
        [
            0.9,
            0.1,
            0.3,
            0.9,
        ],  # High battery, low clean energy, low accuracy, high latency
        [0.5, 0.5, 0.5, 0.5],  # All medium
        [0.1, 0.9, 0.8, 0.2],  # Very low battery, very high clean energy
        [
            0.8,
            0.2,
            0.7,
            0.8,
        ],  # High battery, low clean energy, high accuracy, high latency
    ]

    for i, scenario in enumerate(scenarios):
        logger.debug(f"Testing scenario {i}: {scenario}")

        try:
            input_tensor = torch.tensor([scenario], dtype=torch.float32).to(device)
            model_probs, charge_prob = model(input_tensor)
            logger.debug(
                f"Scenario {i} outputs: model_probs shape={model_probs.shape}, charge_prob shape={charge_prob.shape}"
            )
        except Exception as e:
            logger.error(f"Forward pass failed for scenario {i}: {e}")
            raise

        assert model_probs.shape == (1, 6), (
            f"Scenario {i}: Expected (1, 6), got {model_probs.shape}"
        )
        assert charge_prob.shape == (1,), (
            f"Scenario {i}: Expected (1,), got {charge_prob.shape}"
        )

        # Check for NaN or Inf
        assert not torch.isnan(model_probs).any(), (
            f"Scenario {i}: Model probs contain NaN"
        )
        assert not torch.isnan(charge_prob).any(), (
            f"Scenario {i}: Charge prob contains NaN"
        )
        assert not torch.isinf(model_probs).any(), (
            f"Scenario {i}: Model probs contain Inf"
        )
        assert not torch.isinf(charge_prob).any(), (
            f"Scenario {i}: Charge prob contains Inf"
        )

    logger.info("✓ Forward pass test passed")


def test_backward_pass():
    """Test backward pass and gradient computation."""
    print("Testing backward pass...")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = NeuralController().to(device)
    loss_fn = NeuralLoss()

    # Create test data
    features = torch.randn(3, 4).to(device)
    model_targets = torch.randint(0, 6, (3,)).to(device)
    charge_targets = torch.rand(3).to(device)

    # Forward pass
    model_probs, charge_prob = model(features)

    # Compute loss
    loss = loss_fn(model_probs, charge_prob, model_targets, charge_targets)

    # Backward pass
    loss.backward()

    # Check gradients
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for parameter: {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient for parameter: {name}"
        assert not torch.isinf(param.grad).any(), f"Inf gradient for parameter: {name}"

    print("✓ Backward pass test passed")


def test_loss_function():
    """Test loss function computation."""
    print("Testing loss function...")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    loss_fn = NeuralLoss()

    # Test cases
    test_cases = [
        # Perfect predictions (should have low loss)
        {
            "model_probs": torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).to(device),
            "charge_prob": torch.tensor([0.9]).to(device),
            "model_targets": torch.tensor([0]).to(device),
            "charge_targets": torch.tensor([0.9]).to(device),
            "expected_low": True,
        },
        # Wrong predictions (should have high loss)
        {
            "model_probs": torch.tensor([[0.0, 0.9, 0.1, 0.0, 0.0, 0.0]]).to(device),
            "charge_prob": torch.tensor([0.1]).to(device),
            "model_targets": torch.tensor([0]).to(device),
            "charge_targets": torch.tensor([0.9]).to(device),
            "expected_low": False,
        },
    ]

    losses = []
    for i, case in enumerate(test_cases):
        loss = loss_fn(
            case["model_probs"],
            case["charge_prob"],
            case["model_targets"],
            case["charge_targets"],
        )
        losses.append(loss.item())

        assert not torch.isnan(loss), f"Case {i}: Loss is NaN"
        assert not torch.isinf(loss), f"Case {i}: Loss is Inf"
        assert loss.item() >= 0, f"Case {i}: Loss should be non-negative"

    # Perfect prediction should have lower loss than wrong prediction
    assert losses[0] < losses[1], (
        "Perfect prediction should have lower loss than wrong prediction"
    )

    print("✓ Loss function test passed")


def test_training_step():
    """Test a complete training step."""
    print("Testing training step...")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = NeuralController().to(device)
    loss_fn = NeuralLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Create training scenario
    scenario = {
        "battery_level": 50.0,
        "clean_energy_percentage": 60.0,
        "accuracy_requirement": 0.7,
        "latency_requirement": 15.0,
        "optimal_model": "YOLOv10_M",
        "should_charge": True,
    }

    # Setup model mappings
    model_to_idx = {
        "YOLOv10_N": 0,
        "YOLOv10_S": 1,
        "YOLOv10_M": 2,
        "YOLOv10_B": 3,
        "YOLOv10_L": 4,
        "YOLOv10_X": 5,
    }

    # Extract features
    features = torch.tensor(
        [
            scenario["battery_level"] / 100.0,
            scenario["clean_energy_percentage"] / 100.0,
            scenario["accuracy_requirement"],
            scenario["latency_requirement"] / 30.0,
        ],
        dtype=torch.float32,
    ).to(device)

    target_model_idx = torch.tensor(
        [model_to_idx[scenario["optimal_model"]]], dtype=torch.long
    ).to(device)
    target_charge = torch.tensor([float(scenario["should_charge"])]).to(device)

    # Training step
    model.train()
    optimizer.zero_grad()

    model_probs, charge_prob = model(features.unsqueeze(0))
    loss = loss_fn(model_probs, charge_prob, target_model_idx, target_charge)

    loss.backward()
    optimizer.step()

    assert loss.item() >= 0, "Training loss should be non-negative"
    assert not torch.isnan(loss), "Training loss should not be NaN"

    print("✓ Training step test passed")


def test_model_convergence():
    """Test that model can learn simple patterns."""
    print("Testing model convergence...")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = NeuralController().to(device)
    loss_fn = NeuralLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Create simple training data
    # Rule: if battery < 30% -> charge, if accuracy > 0.8 -> use large model
    train_data = [
        {
            "features": [0.2, 0.5, 0.9, 0.5],
            "model_idx": 4,
            "charge": 1.0,
        },  # Low battery, high accuracy -> large model, charge
        {
            "features": [0.8, 0.3, 0.2, 0.8],
            "model_idx": 0,
            "charge": 0.0,
        },  # High battery, low accuracy -> small model, no charge
        {
            "features": [0.1, 0.9, 0.8, 0.3],
            "model_idx": 3,
            "charge": 1.0,
        },  # Very low battery, high accuracy -> medium-large model, charge
        {
            "features": [0.9, 0.1, 0.1, 0.9],
            "model_idx": 0,
            "charge": 0.0,
        },  # High battery, low accuracy -> small model, no charge
        {
            "features": [0.3, 0.7, 0.7, 0.4],
            "model_idx": 2,
            "charge": 1.0,
        },  # Low battery, medium accuracy -> medium model, charge
    ]

    initial_loss = None
    final_loss = None

    # Train for few epochs
    for epoch in range(50):
        epoch_loss = 0.0
        for data in train_data:
            features = torch.tensor([data["features"]], dtype=torch.float32).to(device)
            model_targets = torch.tensor([data["model_idx"]], dtype=torch.long).to(
                device
            )
            charge_targets = torch.tensor([data["charge"]]).to(device)

            optimizer.zero_grad()
            model_probs, charge_prob = model(features)
            loss = loss_fn(model_probs, charge_prob, model_targets, charge_targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_data)

        if epoch == 0:
            initial_loss = avg_loss
        if epoch == 49:
            final_loss = avg_loss

    # Loss should decrease
    assert initial_loss is not None and final_loss is not None, (
        "Losses should not be None"
    )
    assert final_loss < initial_loss, (
        f"Loss should decrease: initial={initial_loss:.4f}, final={final_loss:.4f}"
    )

    print(
        f"✓ Model convergence test passed (loss: {initial_loss:.4f} -> {final_loss:.4f})"
    )


def main():
    """Run all neural network training tests."""
    logger.info("🧪 Running Neural Network Training Tests")
    logger.info("=" * 50)

    try:
        test_neural_controller_initialization()
        test_forward_pass()
        test_backward_pass()
        test_loss_function()
        test_training_step()
        test_model_convergence()

        logger.info("\n✅ All neural network training tests passed!")
        return 0

    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
