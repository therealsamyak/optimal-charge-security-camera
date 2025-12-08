from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List

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

    def __init__(
        self, clean_energy_series: List[float], task_requirements: List[Dict], config
    ):
        self.config = config
        self.clean_energy_series = clean_energy_series  # Full day data
        self.task_requirements = task_requirements  # Full day requirements
        self.current_timestep = 0

        # Pre-compute optimal schedule for entire day
        self.optimal_schedule = self.solve_full_horizon_milp()

        # Create DP matrix for fast lookup
        self.dp_matrix = self.create_dp_matrix()

    def solve_full_horizon_milp(self):
        """Solve MILP for entire day (288 timesteps at 5-min intervals)"""
        # Load model data from JSON file
        import json
        from pathlib import Path

        profiles_file = Path("model-data/power_profiles.json")
        with open(profiles_file, "r") as f:
            raw_models = json.load(f)

        # Transform profiles to match expected field names
        available_models = {}
        for name, profile in raw_models.items():
            available_models[name] = {
                "accuracy": profile["accuracy"],
                "latency": profile["avg_inference_time_seconds"]
                * 1000,  # Convert to ms
                "power_cost": profile["model_power_mw"],  # Power in mW
            }

        # Use runtime config parameters
        task_interval = self.config.task_interval_seconds  # 5s
        battery_capacity = self.config.battery_capacity_wh  # 5.0Wh
        charge_rate = self.config.charge_rate_watts  # 100W

        # Calculate number of timesteps
        num_timesteps = len(self.clean_energy_series)

        prob = pulp.LpProblem("Full_Horizon_Oracle_Optimization", pulp.LpMaximize)

        # Decision variables for all timesteps
        model_vars = {}
        charge_vars = {}
        battery_vars = {}

        for t in range(num_timesteps):
            model_vars[t] = {
                name: pulp.LpVariable(f"model_{t}_{name}", cat="Binary")
                for name in available_models.keys()
            }
            charge_vars[t] = pulp.LpVariable(f"charge_{t}", cat="Binary")
            battery_vars[t] = pulp.LpVariable(f"battery_{t}", lowBound=0, upBound=100)

        # Objective: maximize clean energy usage
        prob += pulp.lpSum(
            [self.clean_energy_series[t] * charge_vars[t] for t in range(num_timesteps)]
        )

        # Constraints
        for t in range(num_timesteps):
            # Exactly one model per timestep
            prob += pulp.lpSum(model_vars[t].values()) == 1

            # Task requirements
            task_req = self.task_requirements[t]
            for name, specs in available_models.items():
                if specs["accuracy"] < task_req["accuracy"]:
                    prob += model_vars[t][name] == 0
                if specs["latency"] > task_req["latency"]:
                    prob += model_vars[t][name] == 0

            # Battery dynamics
            if t == 0:
                # Initial battery level (start at 50%)
                prob += battery_vars[t] == 50
            else:
                # Battery transition: battery_t = battery_{t-1} + charge_{t-1} * rate - energy_used_{t-1}
                energy_used = pulp.lpSum(
                    [
                        available_models[name]["power_cost"] * model_vars[t - 1][name]
                        for name in available_models.keys()
                    ]
                )
                charge_added = charge_vars[t - 1] * (
                    charge_rate * task_interval / 3600 / battery_capacity * 100
                )
                prob += (
                    battery_vars[t] == battery_vars[t - 1] + charge_added - energy_used
                )

            # Battery bounds
            prob += battery_vars[t] >= 0
            prob += battery_vars[t] <= 100

        # Solve the MILP
        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        # Extract optimal schedule
        optimal_schedule = []
        for t in range(num_timesteps):
            selected_model = None
            for name, var in model_vars[t].items():
                if pulp.value(var) == 1:
                    selected_model = name
                    break

            should_charge = pulp.value(charge_vars[t]) == 1
            optimal_schedule.append((selected_model, should_charge))

        return optimal_schedule

    def create_dp_matrix(self):
        """Create lookup table: timestep → (optimal_model, should_charge)"""
        dp_matrix = {}
        for t, (model, charge) in enumerate(self.optimal_schedule):
            dp_matrix[t] = {"model": model, "charge": charge}
        return dp_matrix

    def select_model(
        self,
        battery_level: float,
        clean_energy_percentage: float,
        user_accuracy_requirement: float,
        user_latency_requirement: float,
        available_models: Dict[str, Dict[str, float]],
    ) -> ModelChoice:
        """Fast DP matrix lookup instead of solving MILP"""
        optimal_decision = self.dp_matrix[self.current_timestep]

        return ModelChoice(
            model_name=optimal_decision["model"],
            should_charge=optimal_decision["charge"],
            reasoning="Optimal full-horizon MILP solution with complete future knowledge",
        )

    def should_charge(self):
        """Return pre-computed charging decision"""
        return self.dp_matrix[self.current_timestep]["charge"]

    def advance_timestep(self):
        """Advance to next timestep"""
        self.current_timestep += 1

    def get_current_timestep(self):
        """Get current timestep index"""
        return self.current_timestep


class CustomController(Controller):
    """Custom controller using neural network for model selection and charging."""

    def __init__(self, weights_file: str = "custom_controller_weights.json"):
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
            if not available_models:
                # No models available - raise error
                raise ValueError("No available models provided to CustomController")
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
