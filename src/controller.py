from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict

import pulp


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
    """Custom controller using trained weights for model selection and charging."""

    def __init__(self, weights_file: str = "results/custom_controller_weights.json"):
        self.weights = {
            "accuracy_weight": 0.5,
            "latency_weight": 0.3,
            "clean_energy_weight": 0.2,
        }
        self.model_weights = {}
        self.charge_weights = None
        self.charge_threshold = 0.0
        self.load_weights(weights_file)

    def load_weights(self, filepath: str):
        """Load trained weights from JSON file."""
        import json

        with open(filepath, "r") as f:
            weights_data = json.load(f)

        if "weights" not in weights_data:
            raise ValueError(f"Missing 'weights' in {filepath}")
        if "model_weights" not in weights_data:
            raise ValueError(f"Missing 'model_weights' in {filepath}")
        if "charge_threshold" not in weights_data:
            raise ValueError(f"Missing 'charge_threshold' in {filepath}")

        self.weights = weights_data["weights"]
        self.model_weights = {
            k: list(v) for k, v in weights_data["model_weights"].items()
        }
        self.charge_weights = weights_data.get("charge_weights")
        self.charge_threshold = weights_data["charge_threshold"]

    def select_model(
        self,
        battery_level: float,
        clean_energy_percentage: float,
        user_accuracy_requirement: float,
        user_latency_requirement: float,
        available_models: Dict[str, Dict[str, float]],
    ) -> ModelChoice:
        # Normalize features
        features = [
            battery_level / 100.0,
            clean_energy_percentage / 100.0,
            user_accuracy_requirement,
            user_latency_requirement / 3000.0,
        ]

        # Calculate model scores using trained weights
        model_scores = {}
        for model in available_models.keys():
            if model not in self.model_weights:
                raise ValueError(f"No trained weights found for model: {model}")

            # Use trained model weights for scoring
            score = sum(f * w for f, w in zip(features, self.model_weights[model]))
            model_scores[model] = score

        selected_model = max(model_scores.keys(), key=lambda x: model_scores[x])

        # Charging decision using trained weights
        if self.charge_weights is None:
            raise ValueError("No charge_weights found in trained model")

        charge_score = sum(f * w for f, w in zip(features, self.charge_weights))
        should_charge = charge_score > 0  # Use 0 as threshold like in training

        return ModelChoice(
            model_name=selected_model,
            should_charge=should_charge,
            reasoning="Custom controller with trained weights",
        )
