"""
Constants and configuration for the security camera simulation system.
Centralizes magic numbers and configuration values.
"""

import logging


class ModelConfig:
    """YOLO model configuration constants."""

    MODELS = [
        "YOLOv10_N",
        "YOLOv10_S",
        "YOLOv10_M",
        "YOLOv10_B",
        "YOLOv10_L",
        "YOLOv10_X",
    ]
    CHARGE_THRESHOLD = 0.5
    MAX_LATENCY_MS = 30.0
    BATTERY_FULL_THRESHOLD = 99.5

    # Neural network architecture
    INPUT_FEATURES = 4  # battery_level, clean_energy_percentage, accuracy_requirement, latency_requirement
    HIDDEN_LAYERS = [128, 64]
    MODEL_OUTPUTS = 6  # Number of YOLO models
    CHARGE_OUTPUTS = 1  # Single charge decision

    # Loss function weights
    ACCURACY_WEIGHT = 0.5
    ENERGY_WEIGHT = 0.5


class TrainingConfig:
    """Training configuration constants."""

    DEFAULT_LEARNING_RATE = 0.001
    DEFAULT_EPOCHS = 1000
    EARLY_STOPPING_PATIENCE = 50
    VALIDATION_SPLIT = 0.2
    TRAIN_SPLIT = 0.7

    # Learning rate scheduler
    SCHEDULER_PATIENCE = 10


class SimulationConfig:
    """Simulation configuration constants."""

    DEFAULT_BATTERY_CAPACITY_WH = 1000.0
    DEFAULT_CHARGE_RATE_WATTS = 10.0
    MIN_BATTERY_LEVEL = 5.0
    MAX_BATTERY_LEVEL = 100.0

    # Task simulation
    DEFAULT_TASKS_PER_SIMULATION = 20
    ENERGY_CONSUMPTION_FACTOR = 0.1


class TestConfig:
    """Test configuration constants."""

    MIN_TEST_SCENARIOS = 1
    MAX_TEST_SCENARIOS = 10
    TEMP_FILE_SUFFIX = ".json"

    # Test locations and timestamps
    TEST_LOCATIONS = [
        "US-CAL-LDWP_2024_5_minute",
        "US-FLA-FPL_2024_5_minute",
        "US-NY-NYIS_2024_5_minute",
    ]
    TEST_TIMESTAMPS = ["2024-01-15 08:00:00", "2024-01-15 20:00:00"]

    # Test battery levels and requirements
    TEST_BATTERY_LEVELS = [20.0, 50.0, 80.0]
    TEST_ACCURACY_REQUIREMENTS = [0.5, 0.8]
    TEST_LATENCY_REQUIREMENTS = [10.0, 25.0]


class LoggingConfig:
    """Logging configuration constants."""

    DEFAULT_LEVEL = logging.INFO
    DEBUG_LEVEL = logging.DEBUG
    FORMAT = "%(levelname)s: %(message)s"

    # Log messages
    MODEL_INITIALIZED = "Model initialized on device: {device}"
    FORWARD_PASS_START = "Forward pass: input shape {input_shape}, device {device}"
    FORWARD_PASS_SUCCESS = "Forward pass: model_probs range [{min:.3f}, {max:.3f}], charge_prob {charge:.3f}"
    TRAINING_START = "Starting training with {epochs} epochs, lr={lr}"
    VALIDATION_METRICS = (
        "Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}"
    )
    EARLY_STOPPING = "Early stopping at epoch {epoch}"
