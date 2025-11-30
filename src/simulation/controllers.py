"""Controller implementations for simulation."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpBinary, LpContinuous, value
from loguru import logger
import hashlib
import json

from src.utils.cache import (
    get_oracle_cache_path,
    load_oracle_cache,
    save_oracle_cache,
)
from src.sensors.model_inference import get_cached_latency


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
        image_path: Optional[str] = None,
    ) -> Optional[str]:
        """Select YOLO model based on current conditions.
        
        Args:
            battery_level: Current battery level as percentage (0-100)
            energy_cleanliness: Current energy cleanliness (0-1)
            model_data: Dictionary of model data
            image_path: Optional path to image file for looking up cached actual latencies
        """
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
        logger.debug(
            f"CustomController initialized with weights: "
            f"accuracy={self.accuracy_weight}, latency={self.latency_weight}, "
            f"energy={self.energy_weight}, battery={self.battery_weight}"
        )

    def select_model(
        self,
        battery_level: float,
        energy_cleanliness: float,
        model_data: Dict[str, Dict[str, float]],
        image_path: Optional[str] = None,
    ) -> Optional[str]:
        """Select model using weighted scoring algorithm.
        
        Filters models by latency SLA before scoring, then scores using normalized features.
        Uses actual measured latencies from inference cache when available, otherwise falls back
        to metadata latencies from model_data.
        
        Args:
            battery_level: Current battery level as percentage (0-100)
            energy_cleanliness: Current energy cleanliness (0-1)
            model_data: Dictionary of model data with metadata latencies
            image_path: Optional path to image file for looking up cached actual latencies
        """
        best_model = None
        best_score = -float("inf")
        valid_models = []
        rejected_models = []

        # First pass: filter out models that violate latency SLA
        # This is a hard constraint - models exceeding latency threshold are not considered
        # Use actual measured latencies when available (from inference cache), otherwise use metadata
        candidate_models = {}
        for model_name, data in model_data.items():
            # Skip if model doesn't meet accuracy threshold
            if data["accuracy"] < self.accuracy_threshold:
                rejected_models.append(
                    (
                        model_name,
                        f"accuracy {data['accuracy']:.3f} < {self.accuracy_threshold}",
                    )
                )
                continue
            
            # Get latency for SLA check: prefer actual measured latency, fall back to metadata
            metadata_latency_ms = data["latency_ms"]
            actual_latency_ms = None
            
            if image_path:
                actual_latency_ms = get_cached_latency(model_name, image_path)
            
            # Use actual latency if available, otherwise use metadata latency
            # Note: metadata latency may be optimistic (e.g., TensorRT optimized),
            # while actual latency reflects real runtime conditions
            latency_for_sla = actual_latency_ms if actual_latency_ms is not None else metadata_latency_ms
            latency_source = "measured" if actual_latency_ms is not None else "metadata"
            
            # Hard filter: models exceeding latency threshold are rejected
            if latency_for_sla > self.latency_threshold:
                logger.debug(
                    f"CustomController: Filtered {model_name} by latency SLA: "
                    f"{latency_for_sla:.2f}ms ({latency_source}) > {self.latency_threshold}ms"
                )
                rejected_models.append(
                    (
                        model_name,
                        f"latency {latency_for_sla:.2f}ms ({latency_source}) > {self.latency_threshold}ms",
                    )
                )
                continue

            # Model passes both thresholds
            # Store the latency used for filtering (for scoring later)
            candidate_data = data.copy()
            candidate_data["latency_ms_for_sla"] = latency_for_sla
            candidate_data["latency_source"] = latency_source
            candidate_models[model_name] = candidate_data
            valid_models.append(model_name)

        # If no models pass the latency filter, fall back to lowest latency model
        if not candidate_models:
            logger.warning(
                f"CustomController: No models meet latency SLA ({self.latency_threshold}ms). "
                f"Rejected {len(rejected_models)} models. Falling back to lowest latency model."
            )
            # Find the model with lowest latency that meets accuracy threshold
            # Use actual latencies when available for fallback selection too
            fallback_candidates = []
            for name, data in model_data.items():
                if data["accuracy"] >= self.accuracy_threshold:
                    # Get actual latency if available, otherwise use metadata
                    fallback_latency = data["latency_ms"]
                    if image_path:
                        actual = get_cached_latency(name, image_path)
                        if actual is not None:
                            fallback_latency = actual
                    fallback_candidates.append((name, data, fallback_latency))
            
            if fallback_candidates:
                fallback_model = min(fallback_candidates, key=lambda x: x[2])
                logger.warning(
                    f"CustomController: Using fallback {fallback_model[0]} "
                    f"(latency: {fallback_model[2]:.2f}ms, "
                    f"accuracy: {fallback_model[1]['accuracy']:.3f})"
                )
                return fallback_model[0]
            else:
                logger.warning(
                    f"CustomController: No valid models found. "
                    f"Rejected {len(rejected_models)} models."
                )
                return None
        
        # Log how many models passed the SLA filter
        logger.debug(
            f"CustomController: {len(candidate_models)} model(s) passed SLA filter "
            f"(from {len(model_data)} total, {len(rejected_models)} rejected)"
        )

        # Second pass: score candidate models using normalized features
        # Use the latency that was used for SLA filtering (actual if available, metadata otherwise)
        # Find max latency among candidates for normalization
        max_latency_ms = max(
            data.get("latency_ms_for_sla", data["latency_ms"]) 
            for data in candidate_models.values()
        )
        # Use a reasonable upper bound for latency normalization (300ms or max observed)
        max_latency_for_norm = max(300.0, max_latency_ms)

        for model_name, data in candidate_models.items():
            # Normalize all features to [0, 1] range
            
            # Accuracy: already in [0, 1] range
            accuracy_norm = data["accuracy"]
            
            # Latency: normalize to [0, 1] where lower is better
            # Use the latency that passed SLA check (actual if available, metadata otherwise)
            latency_for_scoring = data.get("latency_ms_for_sla", data["latency_ms"])
            # Use inverse normalization: (max - current) / max, clamped to [0, 1]
            latency_norm = max(0.0, min(1.0, (max_latency_for_norm - latency_for_scoring) / max_latency_for_norm))
            
            # Energy cleanliness: already in [0, 1] range
            energy_norm = energy_cleanliness
            
            # Battery: normalize from [0, 100] to [0, 1]
            # Option: higher score when battery is low (prioritize energy saving)
            # This means we prefer smaller models when battery is low
            battery_norm = max(0.0, min(1.0, 1.0 - (battery_level / 100.0)))
            # Alternative (if you want higher score when battery is high):
            # battery_norm = max(0.0, min(1.0, battery_level / 100.0))

            # Calculate weighted score
            # All terms are now normalized to [0, 1], so scores are comparable
            accuracy_score = accuracy_norm * self.accuracy_weight
            latency_score = latency_norm * self.latency_weight
            energy_score = energy_norm * self.energy_weight
            battery_score = battery_norm * self.battery_weight

            total_score = accuracy_score + latency_score + energy_score + battery_score

            logger.debug(
                f"CustomController scoring {model_name}: "
                f"acc={accuracy_score:.4f}, lat={latency_score:.4f}, "
                f"energy={energy_score:.4f}, bat={battery_score:.4f}, "
                f"total={total_score:.4f} "
                f"(norms: acc={accuracy_norm:.3f}, lat={latency_norm:.3f}, "
                f"energy={energy_norm:.3f}, bat={battery_norm:.3f})"
            )

            if total_score > best_score:
                best_score = total_score
                best_model = model_name

        if best_model is None:
            logger.warning(
                f"CustomController: No valid models found. "
                f"Rejected {len(rejected_models)} models. "
                f"Thresholds: accuracy>={self.accuracy_threshold}, latency<={self.latency_threshold}ms"
            )
        else:
            selected_data = candidate_models[best_model]
            latency_used = selected_data.get("latency_ms_for_sla", selected_data["latency_ms"])
            latency_source = selected_data.get("latency_source", "metadata")
            logger.debug(
                f"CustomController selected {best_model} with score {best_score:.4f} "
                f"(from {len(valid_models)} valid models, "
                f"latency: {latency_used:.2f}ms {latency_source})"
            )

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
        self.battery_config = config.get("battery", {})
        
        # Get battery capacity in Wh (canonical source: battery.capacity_wh)
        # Default: 4.0 Wh (aligned with config.jsonc default)
        capacity_wh = float(self.battery_config.get("capacity_wh", 4.0))
        
        # Get charging rate in Watts (canonical source: battery.charging_rate_watts)
        # Default: 0.5 W (for 4 Wh battery, gives ~8 hour full charge)
        charging_rate_watts = float(self.battery_config.get("charging_rate_watts", 0.5))
        
        # Derive charging_rate (percent per second) from physical charging power
        # Formula: charging_rate (%/s) = (charging_rate_watts / capacity_wh) * 100 / 3600
        # This converts Watts -> Wh/s -> %/s
        if capacity_wh > 0:
            charging_rate_pct_per_sec = (charging_rate_watts / capacity_wh) * 100.0 / 3600.0
        else:
            charging_rate_pct_per_sec = 0.0035  # Legacy default fallback
        
        # Fallback to legacy charging_rate if new config not available
        legacy_charging_rate = self.battery_config.get("charging_rate")
        if legacy_charging_rate is not None:
            charging_rate_pct_per_sec = float(legacy_charging_rate)
        
        self.charging_rate = charging_rate_pct_per_sec
        self.max_capacity = 100.0  # Always 100% (max is defined by capacity_wh)

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
        """Pre-compute all decisions using MILP optimization.

        Uses caching to avoid re-running expensive MILP optimization
        for the same parameters.
        """
        # Create cache key components
        date_str = start_date.strftime("%Y-%m-%d")
        model_data_hash = hashlib.md5(
            json.dumps(model_data, sort_keys=True, default=str).encode()
        ).hexdigest()

        cache_path = get_oracle_cache_path(
            date=date_str,
            controller_type="oracle",
            accuracy_threshold=self.accuracy_threshold,
            latency_threshold=self.latency_threshold,
            initial_battery=initial_battery,
            model_data_hash=model_data_hash,
        )

        # Check cache
        cached_result = load_oracle_cache(cache_path)
        if cached_result is not None:
            self.decisions = cached_result["decisions"]
            self.initialized = True
            metadata = cached_result.get("metadata", {})
            logger.info(
                f"Oracle controller initialized from cache with {len(self.decisions)} decisions "
                f"(cached on {metadata.get('timestamp', 'unknown')})"
            )
            return

        # Cache miss - run MILP optimization
        logger.info(
            "Initializing Oracle controller with MILP optimization (cache miss)..."
        )

        # Filter valid models that meet thresholds
        model_names = []
        for model_name, data in model_data.items():
            if (
                data["accuracy"] >= self.accuracy_threshold
                and data["latency_ms"] <= self.latency_threshold
            ):
                model_names.append(model_name)

        if not model_names:
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

            # Energy consumed from models (only if a model is selected)
            energy_consumed = lpSum(
                [x[t][m] * model_data[m]["energy_consumption"] for m in model_names]
            )

            # Energy charged (only if charging)
            time_seconds = self.time_step_minutes * 60
            energy_charged = c[t] * self.charging_rate * time_seconds

            # Battery level constraint: b[t] = b[t-1] - energy_consumed + energy_charged
            # This ensures battery balance is maintained
            prob += b[t] == prev_battery - energy_consumed + energy_charged

        # Only one model can be selected per timestep, or charge, or do nothing
        # This ensures: sum(models) + charge <= 1, meaning:
        # - Can select one model (sum=1, c=0)
        # - Can charge (sum=0, c=1)
        # - Can do nothing (sum=0, c=0)
        for t in range(num_steps):
            prob += lpSum([x[t][m] for m in model_names]) + c[t] <= 1

        # Ensure we can't consume energy if battery is insufficient
        # This is already handled by b[t] >= 0 constraint, but we make it explicit
        # The battery balance constraint ensures b[t] >= 0, which prevents over-consumption

        # Battery cannot go below 0
        for t in range(num_steps):
            prob += b[t] >= 0

        # Battery cannot exceed max capacity
        for t in range(num_steps):
            prob += b[t] <= self.max_capacity

        # Solve the problem
        # The problem should always be feasible because:
        # 1. We can always charge (c[t] = 1 for any t)
        # 2. We can always do nothing (all x[t][m] = 0, c[t] = 0)
        # 3. Battery constraints allow any level between 0 and max_capacity
        import time

        start_time = time.time()

        # Use a solver that's more reliable (default solver may vary)
        # PuLP will use the default solver, but we can specify one if needed
        prob.solve()
        solve_time = time.time() - start_time

        # Check solver status (1 = Optimal, -1 = Infeasible, -2 = Unbounded, etc.)
        status_map = {1: "Optimal", -1: "Infeasible", -2: "Unbounded", -3: "Undefined"}
        status_str = status_map.get(prob.status, f"Unknown ({prob.status})")
        logger.info(
            f"MILP optimization completed in {solve_time:.2f}s with status: {status_str}"
        )

        if prob.status != 1:  # Not optimal
            error_msg = (
                f"MILP solver failed with status: {status_str}. "
                f"This indicates a bug in the MILP formulation. "
                f"Problem should always be feasible (can always charge or do nothing)."
            )
            logger.error(error_msg)
            logger.error(
                f"Problem details: {num_steps} timesteps, {len(model_names)} valid models"
            )
            logger.error(
                f"Initial battery: {initial_battery}, Max capacity: {self.max_capacity}"
            )
            raise RuntimeError(error_msg)

        # Extract decisions
        decisions_count = 0
        charging_count = 0
        idle_count = 0
        model_selections = {}

        for t in range(num_steps):
            time_key = time_steps[t]

            # Check if charging
            if value(c[t]) > 0.5:
                self.decisions[time_key] = None  # Charging
                charging_count += 1
            else:
                # Find which model was selected
                selected_model = None
                for m in model_names:
                    if value(x[t][m]) > 0.5:
                        selected_model = m
                        model_selections[m] = model_selections.get(m, 0) + 1
                        break

                if selected_model is None:
                    # Doing nothing (idle)
                    idle_count += 1

                self.decisions[time_key] = selected_model
                if selected_model is not None:
                    decisions_count += 1

        self.initialized = True
        logger.info(
            f"Oracle controller initialized with {len(self.decisions)} decisions "
            f"({decisions_count} model selections, {charging_count} charging periods, {idle_count} idle periods)"
        )
        if model_selections:
            logger.info(f"Model usage distribution: {model_selections}")

        # Verify battery constraints were satisfied
        logger.debug("Verifying battery constraints...")
        for t in range(min(10, num_steps)):  # Check first 10 timesteps
            battery_val = value(b[t])
            if battery_val < 0 or battery_val > self.max_capacity:
                logger.warning(f"Battery constraint violation at t={t}: {battery_val}")

        # Save to cache
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "solve_time": solve_time,
            "status": status_str,
            "decisions_count": decisions_count,
            "charging_count": charging_count,
            "idle_count": idle_count,
            "model_selections": model_selections,
        }
        save_oracle_cache(cache_path, self.decisions, metadata)

    def get_decision_for_time(self, timestamp: datetime) -> Optional[str]:
        """Get pre-computed decision for a specific timestamp.

        Since Oracle optimizes at 5-minute intervals but simulation runs at 10-second intervals,
        we find the nearest 5-minute decision for any given timestamp.
        """
        if not self.initialized:
            return None

        if not self.decisions:
            return None

        # Round to nearest 5-minute interval
        timestamp_rounded = timestamp.replace(second=0, microsecond=0)
        minute_mod = timestamp_rounded.minute % self.time_step_minutes
        if minute_mod >= self.time_step_minutes / 2:
            timestamp_rounded += timedelta(minutes=self.time_step_minutes - minute_mod)
        else:
            timestamp_rounded -= timedelta(minutes=minute_mod)

        # Try exact match first
        decision = self.decisions.get(timestamp_rounded)
        if decision is not None:
            return decision

        # If no exact match, find nearest decision (within 5 minutes)
        # This handles edge cases where timestamp rounding might miss
        min_diff = float("inf")
        nearest_decision = None
        for decision_time, decision_value in self.decisions.items():
            diff = abs((timestamp_rounded - decision_time).total_seconds())
            if diff < min_diff and diff <= self.time_step_minutes * 60:
                min_diff = diff
                nearest_decision = decision_value

        return nearest_decision

    def select_model(
        self,
        battery_level: float,
        energy_cleanliness: float,
        model_data: Dict[str, Dict[str, float]],
        image_path: Optional[str] = None,
    ) -> Optional[str]:
        """Select model using pre-computed MILP decisions."""
        # This method signature is kept for compatibility, but Oracle uses get_decision_for_time
        # The runner will need to call get_decision_for_time instead
        # image_path parameter is ignored for OracleController
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
        logger.debug(
            f"BenchmarkController initialized: "
            f"prefer_largest={self.prefer_largest}, charge_threshold={self.charge_threshold}%"
        )

    def select_model(
        self,
        battery_level: float,
        energy_cleanliness: float,
        model_data: Dict[str, Dict[str, float]],
        image_path: Optional[str] = None,
    ) -> Optional[str]:
        """Always select largest model or charge if battery low."""
        # image_path parameter is ignored for BenchmarkController
        if battery_level < self.charge_threshold:
            logger.debug(
                f"BenchmarkController: Battery {battery_level:.2f}% < threshold {self.charge_threshold}%, "
                f"forcing charge"
            )
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
                    logger.debug(
                        f"BenchmarkController: Selected largest available model {model_name} "
                        f"(battery: {battery_level:.2f}%)"
                    )
                    return model_name

        logger.warning("BenchmarkController: No models available in model_data")
        return None
