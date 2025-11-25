"""Controller implementations for simulation."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpBinary, LpContinuous, value
from loguru import logger


class BaseController(ABC):
    """Base class for all controllers."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.accuracy_threshold = config["accuracy_threshold"]
        self.latency_threshold = config["latency_threshold_ms"]

    @abstractmethod
    def select_model(
        self,
        battery_level: float,
        energy_cleanliness: float,
        model_data: Dict[str, Dict[str, float]],
    ) -> Optional[str]:
        """Select YOLO model based on current conditions."""
        pass


class CustomController(BaseController):
    """Weighted scoring controller balancing multiple factors."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        weights = config["custom_controller_weights"]
        self.accuracy_weight = weights["accuracy_weight"]
        self.latency_weight = weights["latency_weight"]
        self.energy_weight = weights["energy_cleanliness_weight"]
        self.battery_weight = weights["battery_conservation_weight"]

    def select_model(
        self,
        battery_level: float,
        energy_cleanliness: float,
        model_data: Dict[str, Dict[str, float]],
    ) -> Optional[str]:
        """Select model using weighted scoring algorithm."""
        best_model = None
        best_score = -float("inf")

        for model_name, data in model_data.items():
            # Skip if model doesn't meet performance thresholds
            if data["accuracy"] < self.accuracy_threshold:
                continue
            if data["latency_ms"] > self.latency_threshold:
                continue

            # Calculate weighted score
            accuracy_score = data["accuracy"] * self.accuracy_weight
            latency_score = (1.0 / data["latency_ms"]) * self.latency_weight
            energy_score = energy_cleanliness * self.energy_weight
            battery_score = battery_level * self.battery_weight

            total_score = accuracy_score + latency_score + energy_score + battery_score

            if total_score > best_score:
                best_score = total_score
                best_model = model_name

        return best_model


class OracleController(BaseController):
    """MILP-based oracle controller with full future knowledge."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.horizon_hours = config["oracle_controller"]["optimization_horizon_hours"]
        self.time_step_minutes = config["oracle_controller"]["time_step_minutes"]
        self.clean_energy_bonus = config["oracle_controller"][
            "clean_energy_bonus_factor"
        ]
        self.battery_config = config["battery"]
        self.charging_rate = self.battery_config["charging_rate"]
        self.max_capacity = self.battery_config["max_capacity"]

        # Pre-computed decisions (will be set by initialize method)
        self.decisions: Dict[datetime, Optional[str]] = {}
        self.initialized = False

    def initialize(
        self,
        start_date: datetime,
        energy_data: Dict[datetime, float],
        model_data: Dict[str, Dict[str, float]],
        initial_battery: float,
    ) -> None:
        """Pre-compute all decisions using MILP optimization."""
        logger.info("Initializing Oracle controller with MILP optimization...")

        # Filter valid models that meet thresholds
        valid_models = []
        model_names = []
        for model_name, data in model_data.items():
            if (
                data["accuracy"] >= self.accuracy_threshold
                and data["latency_ms"] <= self.latency_threshold
            ):
                valid_models.append(model_name)
                model_names.append(model_name)

        if not valid_models:
            logger.warning("No valid models meet thresholds for Oracle controller")
            self.initialized = True
            return

        # Create time steps (5-minute intervals for 24 hours = 288 steps)
        time_steps = []
        current_time = start_date
        end_time = start_date + timedelta(hours=self.horizon_hours)

        while current_time < end_time:
            time_steps.append(current_time)
            current_time += timedelta(minutes=self.time_step_minutes)

        num_steps = len(time_steps)

        # Create MILP problem
        prob = LpProblem("Oracle_Optimization", LpMaximize)

        # Decision variables
        # x[t][m] = 1 if model m is selected at time t, 0 otherwise
        x = {}
        for t in range(num_steps):
            x[t] = {}
            for m in model_names:
                x[t][m] = LpVariable(f"model_{t}_{m}", cat=LpBinary)

        # c[t] = 1 if charging at time t, 0 otherwise
        c = [LpVariable(f"charge_{t}", cat=LpBinary) for t in range(num_steps)]

        # b[t] = battery level at time t (continuous)
        b = [
            LpVariable(
                f"battery_{t}", lowBound=0, upBound=self.max_capacity, cat=LpContinuous
            )
            for t in range(num_steps)
        ]

        # Objective: maximize total clean energy consumed
        objective_terms = []
        for t in range(num_steps):
            time_key = time_steps[t]
            energy_cleanliness = energy_data.get(time_key, 0.5)

            for m in model_names:
                energy_consumed = model_data[m]["energy_consumption"]
                # Weight by clean energy percentage and bonus factor
                objective_terms.append(
                    x[t][m]
                    * energy_consumed
                    * energy_cleanliness
                    * self.clean_energy_bonus
                )

        prob += lpSum(objective_terms)

        # Constraints

        # Initial battery level
        prob += b[0] == initial_battery

        # Battery balance constraints
        for t in range(num_steps):
            if t == 0:
                prev_battery = initial_battery
            else:
                prev_battery = b[t - 1]

            # Energy consumed from models
            energy_consumed = lpSum(
                [x[t][m] * model_data[m]["energy_consumption"] for m in model_names]
            )

            # Energy charged
            time_seconds = self.time_step_minutes * 60
            energy_charged = c[t] * self.charging_rate * time_seconds

            # Battery level constraint
            prob += b[t] == prev_battery - energy_consumed + energy_charged

        # Only one model can be selected per timestep (or none if charging)
        for t in range(num_steps):
            prob += lpSum([x[t][m] for m in model_names]) + c[t] <= 1

        # Battery cannot go below 0
        for t in range(num_steps):
            prob += b[t] >= 0

        # Battery cannot exceed max capacity
        for t in range(num_steps):
            prob += b[t] <= self.max_capacity

        # Solve the problem
        try:
            prob.solve()
            logger.info(f"MILP optimization status: {prob.status}")

            # Extract decisions
            for t in range(num_steps):
                time_key = time_steps[t]

                # Check if charging
                if value(c[t]) > 0.5:
                    self.decisions[time_key] = None  # Charging
                else:
                    # Find which model was selected
                    selected_model = None
                    for m in model_names:
                        if value(x[t][m]) > 0.5:
                            selected_model = m
                            break
                    self.decisions[time_key] = selected_model

            self.initialized = True
            logger.info(
                f"Oracle controller initialized with {len(self.decisions)} decisions"
            )

        except Exception as e:
            logger.error(f"MILP optimization failed: {e}")
            # Fallback: use greedy approach
            for t in range(num_steps):
                time_key = time_steps[t]
                energy_cleanliness = energy_data.get(time_key, 0.5)
                # Select most efficient model that meets thresholds
                best_model = None
                best_score = -float("inf")
                for m in model_names:
                    score = (
                        energy_cleanliness
                        * self.clean_energy_bonus
                        / model_data[m]["energy_consumption"]
                    )
                    if score > best_score:
                        best_score = score
                        best_model = m
                self.decisions[time_key] = best_model
            self.initialized = True

    def get_decision_for_time(self, timestamp: datetime) -> Optional[str]:
        """Get pre-computed decision for a specific timestamp."""
        if not self.initialized:
            return None

        # Round to nearest 5-minute interval
        timestamp_rounded = timestamp.replace(second=0, microsecond=0)
        minute_mod = timestamp_rounded.minute % self.time_step_minutes
        if minute_mod >= self.time_step_minutes / 2:
            timestamp_rounded += timedelta(minutes=self.time_step_minutes - minute_mod)
        else:
            timestamp_rounded -= timedelta(minutes=minute_mod)

        return self.decisions.get(timestamp_rounded)

    def select_model(
        self,
        battery_level: float,
        energy_cleanliness: float,
        model_data: Dict[str, Dict[str, float]],
    ) -> Optional[str]:
        """Select model using pre-computed MILP decisions."""
        # This method signature is kept for compatibility, but Oracle uses get_decision_for_time
        # The runner will need to call get_decision_for_time instead
        if not self.initialized:
            # Fallback to greedy if not initialized
            valid_models = []
            for model_name, data in model_data.items():
                if (
                    data["accuracy"] >= self.accuracy_threshold
                    and data["latency_ms"] <= self.latency_threshold
                ):
                    valid_models.append((model_name, data["energy_consumption"]))

            if not valid_models:
                return None
            return min(valid_models, key=lambda x: x[1])[0]

        # Return None - runner should use get_decision_for_time instead
        return None


class BenchmarkController(BaseController):
    """Performance-at-all-costs benchmark controller."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.prefer_largest = config["benchmark_controller"]["prefer_largest_model"]
        self.charge_threshold = config["benchmark_controller"]["charge_when_below"]

    def select_model(
        self,
        battery_level: float,
        energy_cleanliness: float,
        model_data: Dict[str, Dict[str, float]],
    ) -> Optional[str]:
        """Always select largest model or charge if battery low."""
        if battery_level < self.charge_threshold:
            return None  # Force charging

        if self.prefer_largest:
            # Find the largest model (YOLOv10-X if available)
            for model_name in [
                "YOLOv10-X",
                "YOLOv10-L",
                "YOLOv10-B",
                "YOLOv10-M",
                "YOLOv10-S",
                "YOLOv10-N",
            ]:
                if model_name in model_data:
                    return model_name

        return None
