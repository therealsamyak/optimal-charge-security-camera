from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict

import pulp
import torch


@dataclass
class ModelChoice:
    """Represents a model selection decision."""

    model_name: str
    should_charge: bool
    reasoning: str


class Controller(ABC):
    """Abstract base class for all controllers."""

    @abstractmethod
    def select_model(
        self,
        battery_level: float,
        clean_energy_percentage: float,
        user_accuracy_requirement: float,
        user_latency_requirement: float,
        available_models: Dict[str, Dict[str, float]],
    ) -> ModelChoice:
        """
        Select which model to use based on current conditions.

        Args:
            battery_level: Current battery percentage (0-100)
            clean_energy_percentage: Current clean energy percentage (0-100)
            user_accuracy_requirement: Minimum accuracy required (0-100)
            user_latency_requirement: Maximum latency allowed (ms)
            available_models: Dict of model_name -> {accuracy, latency, power_cost}

        Returns:
            ModelChoice with selected model and charging decision
        """
        pass


class NaiveWeakController(Controller):
    """Always selects the smallest model (YOLOv10_N)."""

    def select_model(
        self,
        battery_level: float,
        clean_energy_percentage: float,
        user_accuracy_requirement: float,
        user_latency_requirement: float,
        available_models: Dict[str, Dict[str, float]],
    ) -> ModelChoice:
        smallest_model = min(
            available_models.keys(), key=lambda x: available_models[x]["power_cost"]
        )
        return ModelChoice(
            model_name=smallest_model,
            should_charge=battery_level < 20,
            reasoning="Always use smallest model for minimal power",
        )


class NaiveStrongController(Controller):
    """Always selects the largest model (YOLOv10_X)."""

    def select_model(
        self,
        battery_level: float,
        clean_energy_percentage: float,
        user_accuracy_requirement: float,
        user_latency_requirement: float,
        available_models: Dict[str, Dict[str, float]],
    ) -> ModelChoice:
        largest_model = max(
            available_models.keys(), key=lambda x: available_models[x]["power_cost"]
        )
        return ModelChoice(
            model_name=largest_model,
            should_charge=battery_level <= 20,
            reasoning="Always use largest model for maximum accuracy",
        )


class OracleController(Controller):
    """Uses PuLP MILP solver with future knowledge for optimal decisions."""

    def __init__(self, future_energy_data: Dict[int, float], future_tasks: int):
        self.future_energy_data = future_energy_data
        self.future_tasks = future_tasks

    def select_model(
        self,
        battery_level: float,
        clean_energy_percentage: float,
        user_accuracy_requirement: float,
        user_latency_requirement: float,
        available_models: Dict[str, Dict[str, float]],
    ) -> ModelChoice:
        prob = pulp.LpProblem("Oracle_Optimization", pulp.LpMaximize)

        model_vars = {
            name: pulp.LpVariable(f"use_{name}", cat="Binary")
            for name in available_models.keys()
        }
        charge_var = pulp.LpVariable("charge", cat="Binary")

        prob += (
            pulp.lpSum(
                [
                    available_models[name]["accuracy"] * model_vars[name]
                    for name in available_models.keys()
                ]
            )
            - 0.001
            * pulp.lpSum(
                [
                    available_models[name]["latency"] * model_vars[name]
                    for name in available_models.keys()
                ]
            )
            + 0.01 * clean_energy_percentage * charge_var
        )

        prob += pulp.lpSum(model_vars.values()) == 1

        for name, specs in available_models.items():
            if specs["accuracy"] < user_accuracy_requirement:
                prob += model_vars[name] == 0
            if specs["latency"] > user_latency_requirement:
                prob += model_vars[name] == 0

        prob += battery_level + charge_var * 10 <= 100

        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        selected_model = list(available_models.keys())[0]
        for name, var in model_vars.items():
            if pulp.value(var) == 1:
                selected_model = name
                break

        return ModelChoice(
            model_name=selected_model,
            should_charge=pulp.value(charge_var) == 1,
            reasoning="Optimal MILP solution with complete future knowledge",
        )


class CustomController(Controller):
    """Custom controller using neural network for model selection and charging."""

    def __init__(self, weights_file: str = "results/custom_controller_weights.json"):
        # Setup device (Apple Silicon optimization)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Import neural network
        from .neural_controller import NeuralController

        self.model = NeuralController().to(self.device)
        self.model_to_idx = {}
        self.idx_to_model = {}
        self.load_weights(weights_file)

    def load_weights(self, filepath: str):
        """Load trained neural network from JSON file."""
        import json

        with open(filepath, "r") as f:
            weights_data = json.load(f)

        if "model_state_dict" not in weights_data:
            raise ValueError(f"Missing 'model_state_dict' in {filepath}")
        if "model_to_idx" not in weights_data:
            raise ValueError(f"Missing 'model_to_idx' in {filepath}")
        if "idx_to_model" not in weights_data:
            raise ValueError(f"Missing 'idx_to_model' in {filepath}")

        # Load neural network state dict
        state_dict = {
            k: torch.tensor(v) for k, v in weights_data["model_state_dict"].items()
        }
        self.model.load_state_dict(state_dict)

        # Load model mappings
        self.model_to_idx = weights_data["model_to_idx"]
        self.idx_to_model = {int(k): v for k, v in weights_data["idx_to_model"].items()}

    def select_model(
        self,
        battery_level: float,
        clean_energy_percentage: float,
        user_accuracy_requirement: float,
        user_latency_requirement: float,
        available_models: Dict[str, Dict[str, float]],
    ) -> ModelChoice:
        # Normalize features (same as training)
        features = torch.tensor(
            [
                battery_level / 100.0,
                clean_energy_percentage / 100.0,
                user_accuracy_requirement,  # Already 0-1 range
                user_latency_requirement / 30.0,  # Normalize to 30ms max
            ],
            dtype=torch.float32,
        ).to(self.device)

        # Neural network forward pass
        self.model.eval()
        with torch.no_grad():
            model_probs, charge_prob = self.model(features.unsqueeze(0))

            # Get model prediction
            model_idx = int(torch.argmax(model_probs, dim=-1).item())
            selected_model = self.idx_to_model[model_idx]

            # Get charge decision
            should_charge = charge_prob.item() > 0.5

        # Validate selected model is available
        if selected_model not in available_models:
            # Fallback to first available model if prediction is invalid
            selected_model = list(available_models.keys())[0]
            reasoning = "Neural network prediction invalid, using fallback model"
        else:
            reasoning = "Neural network model selection"

        # Never charge if battery is already full (>= 99.5%)
        if battery_level >= 99.5:
            should_charge = False

        return ModelChoice(
            model_name=selected_model,
            should_charge=should_charge,
            reasoning=reasoning,
        )
