#!/usr/bin/env python3
"""
Finite-horizon tree search algorithm for battery-powered model selection.
Enumerates all possible sequences of model selection and charging decisions
to evaluate aggregate energy usage and decision outcomes.
"""

import json
import csv
import dataclasses
import argparse
import logging
import time
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime
import hashlib


@dataclasses.dataclass
class TreeNode:
    """Represents a node in decision tree."""

    battery_level_wh: float  # Actual battery energy in Wh
    timestep: int
    action_sequence: List[Tuple[str, bool]]
    agg_clean_energy: float  # Wh (total amount)
    agg_dirty_energy: float  # Wh (total amount)
    decision_history: List[Dict]  # success/small_miss/large_miss
    parent: Optional["TreeNode"] = None

    def get_clean_energy_percentage(self) -> float:
        """Calculate clean energy percentage."""
        total = self.agg_clean_energy + self.agg_dirty_energy
        return (self.agg_clean_energy / total * 100) if total > 0 else 0.0

    def get_decision_counts(self) -> Dict[str, int]:
        """Count different decision outcomes."""
        counts = {"success": 0, "small_miss": 0, "large_miss": 0}
        for decision in self.decision_history:
            outcome = decision["outcome"]
            if outcome in counts:
                counts[outcome] += 1
        return counts

    def get_total_energy(self) -> float:
        """Calculate total energy consumed."""
        return self.agg_clean_energy + self.agg_dirty_energy


class TreeSearch:
    """Main tree search implementation."""

    def __init__(
        self,
        config_path: str,
        location: str = "CA",
        season: str = "winter",
        parallel: bool = False,
    ):
        self.config_path = config_path
        self.location = location
        self.season = season
        self.parallel = parallel

        # Load configuration and data
        self.config = self._load_config()
        self.models = self._load_models()
        energy_data_tuple = self._load_energy_data()
        self.energy_data = energy_data_tuple[0]
        self.energy_data_date = energy_data_tuple[1]

        # Calculate search parameters
        self.horizon = int(
            (self.config["simulation"]["duration_days"] * 24 * 3600)
            / self.config["simulation"]["task_interval_seconds"]
        )

        # Setup logging with season prefix
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Performance tracking
        self.nodes_explored = 0
        self.nodes_pruned = 0
        self.leaves_found = 0
        self.children_found = 0
        self.children_pruned = 0
        self.start_time = None
        self.last_progress_time = None

        # Memoization cache
        self.state_cache = {}
        self.cache_hits = 0

        # Beam search configuration
        self.beam_config = self.config.get("beam_search", {})
        self.beam_enabled = self.beam_config.get("enabled", False)
        self.beam_width = self.beam_config.get("beam_width_per_bucket", 10)
        self.bucket_sizes = self.beam_config.get(
            "bucket_sizes",
            {
                "success": 10,
                "success_small_miss": 10,
                "most_clean_energy": 10,
                "least_total_energy": 10,
            },
        )

        # Beam search performance tracking
        self.frontier_sizes = []
        self.bucket_utilization = {bucket: [] for bucket in self.bucket_sizes.keys()}

        # Beam reset interval
        self.beam_reset_interval = self.beam_config.get("beam_reset_interval", 5)

        # Parallel configuration
        self.parallel_config = self.config.get("parallel", {})
        self.leaf_target = self.parallel_config.get("leaf_target", 100)

    def _log_with_season(self, message: str, level: str = "info"):
        """Log message with season prefix."""
        prefixed_message = f"[{self.season.upper()}] {message}"
        if level == "info":
            self.logger.info(prefixed_message)
        elif level == "warning":
            self.logger.warning(prefixed_message)
        elif level == "error":
            self.logger.error(prefixed_message)
        elif level == "debug":
            self.logger.debug(prefixed_message)

    def _load_config(self) -> Dict:
        """Load configuration from JSONC file."""
        with open(self.config_path, "r") as f:
            content = f.read()
            # Remove JSONC comments
            lines = [
                line
                for line in content.split("\n")
                if not line.strip().startswith("//")
            ]
            content = "\n".join(lines)
            return json.loads(content)

    def _load_models(self) -> Dict[str, Dict]:
        """Load model profiles from JSON file."""
        models_path = Path("model-data/power_profiles.json")
        with open(models_path, "r") as f:
            raw_models = json.load(f)

        # Transform to expected format
        models = {}
        for name, profile in raw_models.items():
            models[name] = {
                "accuracy": profile["accuracy"],
                "latency": profile["avg_inference_time_seconds"],
                "energy_per_inference_mwh": profile["energy_per_inference_mwh"],
                "power_mw": profile["model_power_mw"],
            }
        return models

    def _load_energy_data(self) -> Tuple[Dict[int, float], str]:
        """Load and interpolate energy data for location."""
        # Map location to filename
        location_files = {
            "CA": "US-CAL-LDWP_2024_5_minute.csv",
            "FL": "US-FLA-FPL_2024_5_minute.csv",
            "NW": "US-NW-PSEI_2024_5_minute.csv",
            "NY": "US-NY-NYIS_2024_5_minute.csv",
        }

        filename = location_files.get(self.location)
        if not filename:
            raise ValueError(f"Unknown location: {self.location}")

        # Load CSV data
        energy_path = Path(f"energy-data/{filename}")
        energy_data = []

        with open(energy_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                energy_data.append(
                    {
                        "timestamp": row["Datetime (UTC)"],
                        "clean_pct": float(
                            row.get("Carbon-free energy percentage (CFE%)", 50.0)
                        ),
                    }
                )

        # Extract sample date from first data point
        sample_date = energy_data[0]["timestamp"].split("T")[0] if energy_data else ""

        # Filter by season and interpolate
        season_data = self._filter_by_season(energy_data)
        return self._interpolate_energy_data(season_data), sample_date

    def _filter_by_season(self, data: List[Dict]) -> List[Dict]:
        """Filter energy data by season."""
        season_months = {
            "winter": [12, 1, 2],
            "spring": [3, 4, 5],
            "summer": [6, 7, 8],
            "fall": [9, 10, 11],
        }

        months = season_months.get(self.season, [1, 2, 3])
        filtered = []

        for entry in data:
            try:
                month = datetime.fromisoformat(entry["timestamp"]).month
                if month in months:
                    filtered.append(entry)
            except (KeyError, ValueError):
                continue

        return filtered

    def _interpolate_energy_data(self, data: List[Dict]) -> Dict[int, float]:
        """Interpolate 5-minute energy data to task intervals."""
        if not data:
            raise RuntimeError(
                f"No energy data available for {self.location} {self.season}"
            )

        energy_map = {}
        data.sort(key=lambda x: x["timestamp"])

        # Convert to seconds and interpolate
        for i in range(len(data) - 1):
            current_time = datetime.fromisoformat(data[i]["timestamp"])
            next_time = datetime.fromisoformat(data[i + 1]["timestamp"])

            current_seconds = (
                current_time - current_time.replace(hour=0, minute=0, second=0)
            ).total_seconds()
            next_seconds = (
                next_time - next_time.replace(hour=0, minute=0, second=0)
            ).total_seconds()

            current_clean = data[i]["clean_pct"]
            next_clean = data[i + 1]["clean_pct"]

            # Interpolate for each task interval
            task_interval = self.config["simulation"]["task_interval_seconds"]
            steps = int((next_seconds - current_seconds) / task_interval)

            for step in range(steps):
                t = step / steps if steps > 0 else 0
                interpolated_clean = current_clean + t * (next_clean - current_clean)
                timestamp = current_seconds + step * task_interval
                energy_map[int(timestamp)] = interpolated_clean

        return energy_map

    def _classify_models(self, node: TreeNode) -> Dict[str, List[str]]:
        """Classify models into qualified, fallback, and unavailable categories."""
        battery_level_wh = node.battery_level_wh
        task_req = {
            "accuracy": self.config["simulation"]["user_accuracy_requirement"] / 100.0,
            "latency": self.config["simulation"]["user_latency_requirement"],
        }

        qualified = []  # Meet requirements AND have sufficient battery
        fallback = []  # Can run but don't meet requirements
        unavailable = []  # Insufficient battery

        for model_name, model_spec in self.models.items():
            meets_accuracy = model_spec["accuracy"] >= task_req["accuracy"]
            meets_latency = model_spec["latency"] <= task_req["latency"]
            energy_needed_wh = model_spec["energy_per_inference_mwh"] / 1000.0
            has_battery = battery_level_wh >= energy_needed_wh

            if not has_battery:
                unavailable.append(model_name)
            elif meets_accuracy and meets_latency:
                qualified.append(model_name)
            else:
                fallback.append(model_name)

        # Add debug logging for model classification
        self._log_with_season(
            f"Model classification: qualified={len(qualified)} {qualified}, "
            f"fallback={len(fallback)} {fallback}, "
            f"unavailable={len(unavailable)} {unavailable}, "
            f"battery_level={battery_level_wh:.6f}Wh",
            "debug",
        )

        return {
            "qualified": qualified,
            "fallback": fallback,
            "unavailable": unavailable,
        }

    def _should_allow_idle(self, node: TreeNode) -> bool:
        """Determine if IDLE action should be allowed based on strict logical rules."""
        battery_level_wh = node.battery_level_wh
        battery_capacity_wh = self.config["battery"]["capacity_wh"]

        # Battery empty → must charge
        if battery_level_wh <= 0:
            return True

        # Battery full → no charging needed, IDLE is illogical
        if battery_level_wh >= battery_capacity_wh:
            return False

        # Check if any model can run with current battery
        for model_name, model_spec in self.models.items():
            energy_needed_wh = model_spec["energy_per_inference_mwh"] / 1000.0
            if battery_level_wh >= energy_needed_wh:
                return False  # At least one model can run

        # No models possible → can idle+charge
        return True

    def _any_model_possible(self, node: TreeNode) -> bool:
        """Check if any model can be executed from current state."""
        classification = self._classify_models(node)
        # Any model is possible if there are qualified or fallback models available
        return (
            len(classification["qualified"]) > 0 or len(classification["fallback"]) > 0
        )

    def _meets_requirements(self, model_name: str) -> bool:
        """Check if model meets user accuracy and latency requirements."""
        model_spec = self.models[model_name]
        task_req = {
            "accuracy": self.config["simulation"]["user_accuracy_requirement"] / 100.0,
            "latency": self.config["simulation"]["user_latency_requirement"],
        }
        return (
            model_spec["accuracy"] >= task_req["accuracy"]
            and model_spec["latency"] <= task_req["latency"]
        )

    def _has_enough_battery(self, model_name: str, battery_level_wh: float) -> bool:
        """Check if battery can run model."""
        energy_needed_wh = self.models[model_name]["energy_per_inference_mwh"] / 1000.0
        return battery_level_wh >= energy_needed_wh

    def _generate_charging_actions(
        self, model_name: str, node: TreeNode
    ) -> List[Tuple[str, bool]]:
        """Generate (model, False) and (model, True) if charging possible."""
        actions = [(model_name, False)]  # Without charging
        battery_capacity_wh = self.config["battery"]["capacity_wh"]
        if node.battery_level_wh < battery_capacity_wh:
            actions.append((model_name, True))  # With charging
        return actions

    def _get_valid_actions(self, node: TreeNode) -> List[Tuple[str, bool]]:
        """Get valid actions for current node using priority-based model selection."""
        battery_level_wh = node.battery_level_wh
        battery_capacity_wh = self.config["battery"]["capacity_wh"]

        # Pruning: battery = 0% → only idle+charging
        if battery_level_wh <= 0:
            return [("IDLE", True)]

        # Priority 1: Find qualifying models that can run without negative battery
        qualifying_runnable = []
        for model_name in self.models.keys():
            if self._meets_requirements(model_name):
                energy_needed = (
                    self.models[model_name]["energy_per_inference_mwh"] / 1000.0
                )
                if (
                    battery_level_wh >= energy_needed
                ):  # Can run without negative battery
                    qualifying_runnable.append(model_name)

        if qualifying_runnable:
            # Generate branches for ALL qualified models
            all_actions = []
            for model in qualifying_runnable:
                all_actions.extend(self._generate_charging_actions(model, node))
            self._log_with_season(
                f"Oracle branching: {len(qualifying_runnable)} qualified models → {len(all_actions)} actions",
                "debug",
            )
            return all_actions

        # Priority 2: Find fallback models that can run without negative battery
        fallback_runnable = []
        for model_name in self.models.keys():
            if not self._meets_requirements(model_name):
                energy_needed = (
                    self.models[model_name]["energy_per_inference_mwh"] / 1000.0
                )
                if (
                    battery_level_wh >= energy_needed
                ):  # Can run without negative battery
                    fallback_runnable.append(model_name)

        if fallback_runnable:
            # Select LARGEST by ACCURACY fallback model
            best_fallback = max(
                fallback_runnable,
                key=lambda x: self.models[x]["accuracy"],
            )
            self._log_with_season(
                f"Oracle selection: {len(fallback_runnable)} fallback models available, "
                f"selected highest accuracy: {best_fallback}",
                "debug",
            )
            return self._generate_charging_actions(best_fallback, node)

        # Priority 3: No models can run → IDLE + charge
        if battery_level_wh < battery_capacity_wh:
            self._log_with_season(
                f"Oracle selection: No models can run (battery: {battery_level_wh:.6f}Wh), "
                f"forcing IDLE + charge",
                "debug",
            )
            return [("IDLE", True)]
        else:
            # Battery full but no models can run (shouldn't happen with current models)
            self._log_with_season(
                "Oracle selection: Battery full but no models can run, no actions available",
                "debug",
            )
            return []

    def _get_naive_action(self, node: TreeNode) -> Tuple[str, bool]:
        """Get naive action: largest model + simple charging rule."""
        battery_level_wh = node.battery_level_wh

        # Pruning: battery = 0% → only idle+charging
        if battery_level_wh <= 0:
            return ("IDLE", True)

        # Try models from largest to smallest (by energy consumption)
        models_by_energy = sorted(
            self.models.keys(),
            key=lambda x: self.models[x]["energy_per_inference_mwh"],
            reverse=True,
        )

        for model_name in models_by_energy:
            if self._has_enough_battery(model_name, battery_level_wh):
                meets_req = self._meets_requirements(model_name)
                should_charge = not meets_req  # Charge ONLY when not optimal
                return (model_name, should_charge)

        # No models can run → IDLE + charge
        return ("IDLE", True)

    def _apply_action(
        self, node: TreeNode, action: Tuple[str, bool], clean_pct: float
    ) -> Optional[TreeNode]:
        """Apply action to create new node state with proper energy accounting order."""
        model_name, should_charge = action
        task_interval = self.config["simulation"]["task_interval_seconds"]
        battery_capacity_wh = self.config["battery"]["capacity_wh"]
        charge_rate_w = (
            self.config["battery"]["capacity_wh"]
            / self.config["battery"]["charge_rate_hours"]
        )  # Wh per hour
        charge_rate_w = charge_rate_w * (
            task_interval / 3600.0
        )  # Convert to Wh per task interval

        # Calculate current battery energy
        current_battery_wh = node.battery_level_wh

        # Initialize new state
        new_clean_energy = node.agg_clean_energy
        new_dirty_energy = node.agg_dirty_energy

        # Calculate energy changes
        charge_energy_wh = 0.0
        model_energy_wh = 0.0

        # Handle charging energy calculation
        if should_charge:
            charge_energy_wh = (
                charge_rate_w  # Already converted to Wh per task interval
            )
            space_available = battery_capacity_wh - current_battery_wh
            actual_charge = min(charge_energy_wh, space_available)

            # Debug logging for charging
            self._log_with_season(
                f"Charging: raw={charge_energy_wh:.8f}Wh, space={space_available:.8f}Wh, actual={actual_charge:.8f}Wh",
                "debug",
            )
        else:
            actual_charge = 0.0

        charge_energy_wh = actual_charge

        # Handle model energy calculation
        if model_name != "IDLE":
            model_spec = self.models[model_name]
            model_energy_wh = (
                model_spec["energy_per_inference_mwh"] / 1000.0
            )  # mWh to Wh

        # Validate net battery change BEFORE applying
        if not self._validate_net_battery_change(
            current_battery_wh, charge_energy_wh, model_energy_wh
        ):
            return None  # Action would violate battery constraints

        # Apply energy changes atomically
        net_battery_wh = current_battery_wh + charge_energy_wh - model_energy_wh

        # Add charging energy to aggregates
        if charge_energy_wh > 0:
            clean_added = charge_energy_wh * (clean_pct / 100.0)
            dirty_added = charge_energy_wh * ((100 - clean_pct) / 100.0)
            new_clean_energy += clean_added
            new_dirty_energy += dirty_added

        # Handle model usage with corrected decision outcome logic
        decision_outcome = "success"
        if model_name == "IDLE":
            # Idle action (always with charging) = large miss
            decision_outcome = "large_miss"
        else:
            # Model energy does NOT add to dirty energy (only charging affects energy accounting)
            # Model usage only affects battery level, not clean/dirty energy tracking

            # Classify available models BEFORE determining outcome
            classification = self._classify_models(node)

            # Check if chosen model is in qualified list and battery >= 0
            is_qualified = model_name in classification["qualified"]
            is_fallback = model_name in classification["fallback"]

            if is_qualified and net_battery_wh >= 0:
                decision_outcome = "success"
            elif is_fallback and net_battery_wh >= 0:
                decision_outcome = "small_miss"
            else:
                # Model chosen but battery would be negative or model unavailable
                decision_outcome = "large_miss"

            # Add debug logging
            self._log_with_season(
                f"Classification: qualified={len(classification['qualified'])}, "
                f"fallback={len(classification['fallback'])}, "
                f"chosen={model_name}, outcome={decision_outcome}, "
                f"battery_after={net_battery_wh:.6f}Wh",
                "debug",
            )

        # Create new node
        new_node = TreeNode(
            battery_level_wh=net_battery_wh,
            timestep=node.timestep + 1,
            action_sequence=node.action_sequence + [action],
            agg_clean_energy=new_clean_energy,
            agg_dirty_energy=new_dirty_energy,
            decision_history=node.decision_history
            + [
                {
                    "timestep": node.timestep,
                    "model": model_name,
                    "charged": should_charge,
                    "outcome": decision_outcome,
                    "charge_energy_used": charge_energy_wh,
                    "clean_energy_pct": clean_pct,
                }
            ],
            parent=node,
        )

        return new_node

    def _get_state_key(self, node: TreeNode, clean_pct: float) -> str:
        """Generate cache key for state memoization."""
        state_data = (
            round(node.battery_level_wh, 6),
            node.timestep,
            round(node.agg_clean_energy, 6),
            round(node.agg_dirty_energy, 6),
            round(clean_pct, 2),
        )
        return hashlib.md5(str(state_data).encode()).hexdigest()

    def _expand_sequential_to_leaf_limit(self, root: TreeNode) -> List[TreeNode]:
        """Expand tree sequentially without pruning until leaf target reached."""
        current_level = [root]
        current_depth = 0

        self._log_with_season(
            f"Starting sequential expansion to leaf target: {self.leaf_target}"
        )

        while current_depth < self.horizon:
            # Estimate next level size (no pruning)
            estimated_next_size = sum(
                len(self._get_valid_actions(node)) for node in current_level
            )

            if len(current_level) + estimated_next_size > self.leaf_target:
                self._log_with_season(
                    f"Stopping sequential expansion at depth {current_depth}: "
                    f"current_leaves={len(current_level)}, "
                    f"estimated_next={estimated_next_size}, "
                    f"target={self.leaf_target}"
                )
                break  # Stop here to avoid exceeding target

            # Expand to next level (no pruning)
            next_level = []
            for node in current_level:
                self.nodes_explored += 1

                # Get clean energy percentage for current timestep
                timestamp = (
                    current_depth * self.config["simulation"]["task_interval_seconds"]
                )
                clean_pct = self.energy_data.get(int(timestamp % 86400), 50.0)

                # Get all valid actions (no pruning)
                valid_actions = self._get_valid_actions(node)

                # Add children to next level
                for action in valid_actions:
                    child = self._apply_action(node, action, clean_pct)
                    if child:
                        self.children_found += 1
                        next_level.append(child)
                    else:
                        self.children_pruned += 1

            current_level = next_level
            current_depth += 1

            # Progress logging
            if current_depth % 10 == 0:
                self._log_with_season(
                    f"Sequential expansion progress: depth={current_depth}, "
                    f"leaves={len(current_level)}"
                )

        self._log_with_season(
            f"Sequential expansion complete: {len(current_level)} leaf nodes at depth {current_depth}"
        )
        return current_level

    def _expand_subtree_worker(self, worker_args: Tuple) -> List[TreeNode]:
        """Worker function for parallel beam search expansion."""
        node, config_path, location, season = worker_args

        # Create worker-specific TreeSearch instance
        worker_search = TreeSearch(config_path, location, season, parallel=False)
        worker_search.nodes_explored = 0
        worker_search.nodes_pruned = 0
        worker_search.leaves_found = 0
        worker_search.children_found = 0
        worker_search.children_pruned = 0

        # Enable beam search for worker
        worker_search.config["beam_search"]["enabled"] = True

        # Use same beam sizes as config for parallel workers
        worker_search.bucket_sizes = self.bucket_sizes.copy()

        # Continue expansion from this leaf node using beam search
        return worker_search._beam_search_expand(node)

    def _run_naive_search_worker(self, worker_args: Tuple) -> TreeNode:
        """Naive search worker for parallel execution."""
        config_path, location, season = worker_args

        # Create dedicated naive search instance
        naive_search = TreeSearch(config_path, location, season, parallel=False)

        # Create root node
        current_node = TreeNode(
            battery_level_wh=naive_search.config["battery"]["capacity_wh"],
            timestep=0,
            action_sequence=[],
            agg_clean_energy=0.0,
            agg_dirty_energy=0.0,
            decision_history=[],
        )

        # Run naive greedy simulation for full horizon
        for timestep in range(naive_search.horizon):
            # Get clean energy percentage for current timestep
            timestamp = (
                timestep * naive_search.config["simulation"]["task_interval_seconds"]
            )
            clean_pct = naive_search.energy_data.get(int(timestamp % 86400), 50.0)

            # Get naive action and apply it
            action = naive_search._get_naive_action(current_node)
            next_node = naive_search._apply_action(current_node, action, clean_pct)

            if not next_node:
                # Safety check - stop if action fails
                naive_search.logger.warning(
                    f"Naive search failed at timestep {timestep}"
                )
                break

            current_node = next_node

        naive_search._log_with_season(
            f"Naive search completed: {current_node.timestep} timesteps"
        )
        return current_node

    def _expand_hybrid_parallel(self, root: TreeNode) -> List[TreeNode]:
        """Hybrid parallel expansion: sequential to leaf target, then parallel beam search."""
        # Phase 1: Sequential expansion to leaf limit
        leaf_nodes = self._expand_sequential_to_leaf_limit(root)

        if not leaf_nodes:
            self.logger.warning("No leaf nodes found in sequential phase")
            return []

        # Phase 2: Parallel beam processing
        max_workers = min(len(leaf_nodes), self.config["parallel"]["max_workers"])
        worker_args = [
            (node, self.config_path, self.location, self.season) for node in leaf_nodes
        ]

        self._log_with_season(
            f"Starting parallel beam processing: {len(leaf_nodes)} leaf nodes, {max_workers} workers"
        )

        all_leaves = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_args = {
                executor.submit(self._expand_subtree_worker, args): args
                for args in worker_args
            }

            for future in as_completed(future_to_args):
                try:
                    worker_leaves = future.result()  # No timeout
                    all_leaves.extend(worker_leaves)
                    self.leaves_found += len(worker_leaves)
                except Exception as e:
                    self.logger.error(f"Worker failed: {e}")
                    # Continue with other workers

        self._log_with_season(
            f"Parallel beam processing complete: {len(all_leaves)} total leaves from {len(leaf_nodes)} workers"
        )

        # Aggregate final beams using all worker data
        self._log_with_season("Aggregating final beams from all worker results...")
        aggregated_leaves = self._assign_to_buckets(all_leaves)

        self._log_with_season(
            f"Final aggregation complete: {len(aggregated_leaves)} leaves after beam aggregation"
        )
        return aggregated_leaves

    def _beam_search_expand(self, root: TreeNode) -> List[TreeNode]:
        """Pareto bucketed beam search for efficient tree exploration."""
        if not self.config.get("beam_search", {}).get("enabled", False):
            # Fall back to original tree search if beam search disabled
            return self._expand_tree_fallback(root, 0)

        self._log_with_season("Starting Pareto bucketed beam search")

        # Initialize frontier with root node
        frontier = [root]

        # Track beam search statistics
        beam_stats = {
            "avg_frontier_size": 0,
            "total_states_generated": 0,
            "total_states_pruned": 0,
            "bucket_utilization": {
                "success": 0,
                "success_small_miss": 0,
                "most_clean_energy": 0,
                "least_total_energy": 0,
            },
        }

        for timestep in range(self.horizon):
            if not frontier:
                self._log_with_season(
                    f"All buckets empty at timestep {timestep}", "error"
                )
                break

            beam_stats["avg_frontier_size"] += len(frontier)

            # Get clean energy percentage for current timestep
            timestamp = timestep * self.config["simulation"]["task_interval_seconds"]
            clean_pct = self.energy_data.get(int(timestamp % 86400), 50.0)

            # Generate all possible next states from current frontier
            next_states = []
            for node in frontier:
                self.nodes_explored += 1
                valid_actions = self._get_valid_actions(node)

                for action in valid_actions:
                    child = self._apply_action(node, action, clean_pct)
                    if child and self._is_valid_state(child):
                        next_states.append(child)
                        self.children_found += 1
                    else:
                        self.children_pruned += 1

            beam_stats["total_states_generated"] += len(next_states)

            # Apply beam reset logic
            if (
                self.beam_reset_interval > 0
                and timestep % self.beam_reset_interval == 0
            ):
                # Reset: assign all current states to buckets
                frontier = self._assign_to_buckets(next_states)
                beam_stats["total_states_pruned"] += len(next_states) - len(frontier)
            else:
                # Normal expansion: keep all states
                frontier = next_states

            # Progress logging
            if timestep % 50 == 0 or timestep == self.horizon - 1:
                self._log_with_season(
                    f"[{datetime.now().strftime('%H:%M:%S')}] Timestep {timestep}/{self.horizon}: "
                    f"frontier_size={len(frontier)}, "
                    f"states_generated={len(next_states)}, "
                    f"states_pruned={len(next_states) - len(frontier)}"
                )

        # Final bucket assignment if last reset wasn't at horizon
        if (
            self.beam_reset_interval > 0
            and (self.horizon - 1) % self.beam_reset_interval != 0
        ):
            self._log_with_season("Applying final bucket assignment...")
            frontier = self._assign_to_buckets(frontier)

        # Final statistics
        if self.horizon > 0:
            beam_stats["avg_frontier_size"] /= self.horizon

        self._log_with_season("Beam search complete:")
        self._log_with_season(
            f"  Average frontier size: {beam_stats['avg_frontier_size']:.1f}"
        )
        self._log_with_season(
            f"  Total states generated: {beam_stats['total_states_generated']:,}"
        )
        self._log_with_season(
            f"  Total states pruned: {beam_stats['total_states_pruned']:,}"
        )
        self._log_with_season(f"  Final frontier size: {len(frontier)}")

        # Store beam search metadata
        self.beam_stats = beam_stats

        return frontier

    def _expand_tree_fallback(self, node: TreeNode, depth: int) -> List[TreeNode]:
        """Fallback to original tree expansion for compatibility."""
        # Safety checks to prevent infinite loops
        max_safe_nodes = float("inf")  #
        max_safe_runtime = float("inf")  #
        max_safe_depth = float("inf")  # Prevent excessive depth

        # Use BFS with explicit stack to avoid recursion
        stack = [(node, depth)]
        all_leaves = []

        while stack:
            current_node, current_depth = stack.pop()
            self.nodes_explored += 1

            # Safety checks
            if self.nodes_explored >= max_safe_nodes:
                self._log_with_season(
                    f"SAFETY: Node limit reached ({max_safe_nodes:,})", "warning"
                )
                break
            if current_depth >= max_safe_depth:
                self._log_with_season(
                    f"SAFETY: Depth limit reached ({max_safe_depth:,})", "warning"
                )
                break

            # Progress logging every 10000 nodes
            if self.nodes_explored % 10000 == 0:
                current_time = time.time()
                if self.last_progress_time:
                    elapsed = current_time - self.last_progress_time
                    rate = 10000 / elapsed if elapsed > 0 else 0
                    progress_pct = (
                        (current_depth / self.horizon) * 100 if self.horizon > 0 else 0
                    )

                    self._log_with_season(
                        f"[{datetime.now().strftime('%H:%M:%S')}] Progress: {self.nodes_explored:,} nodes explored, "
                        f"{self.nodes_pruned:,} nodes pruned, "
                        f"{len(all_leaves):,} leaves found, "
                        f"depth: {current_depth}/{self.horizon} ({progress_pct:.1f}%), "
                        f"stack_size: {len(stack)}, "
                        f"rate: {rate:.0f} nodes/sec"
                    )
                self.last_progress_time = current_time

            # Runtime safety check
            total_runtime = time.time() - (self.start_time or 0)
            if total_runtime >= max_safe_runtime:
                self._log_with_season(
                    f"SAFETY: Runtime limit reached ({max_safe_runtime}s)", "warning"
                )
                break

            # Check if we've reached horizon
            if current_depth >= self.horizon:
                self.leaves_found += 1
                all_leaves.append(current_node)
                continue

            # Get clean energy percentage for current timestep
            timestamp = (
                current_depth * self.config["simulation"]["task_interval_seconds"]
            )
            clean_pct = self.energy_data.get(int(timestamp % 86400), 50.0)

            # Get valid actions (with pruning)
            valid_actions = self._get_valid_actions(current_node)

            if not valid_actions:
                self.nodes_pruned += 1
                continue  # No valid actions, prune this branch

            # Add children to stack (reverse order for BFS-like behavior)
            for action in reversed(valid_actions):
                child = self._apply_action(current_node, action, clean_pct)
                if child:
                    self.children_found += 1
                    stack.append((child, current_depth + 1))
                else:
                    self.children_pruned += 1

            # Safety check after processing current node
            if (
                self.nodes_explored >= max_safe_nodes
                or total_runtime >= max_safe_runtime
            ):
                self._log_with_season(
                    "SAFETY: Tree expansion terminated due to safety limits", "warning"
                )
                break

        return all_leaves

    def _bucket_most_clean_energy(self, states: List[TreeNode]) -> List[TreeNode]:
        """Bucket: Most Clean Energy - highest clean energy percentage, random tiebreak."""
        bucket_size = self.config["beam_search"]["bucket_sizes"]["most_clean_energy"]

        # Sort by clean energy percentage
        sorted_states = sorted(
            states, key=lambda x: x.get_clean_energy_percentage(), reverse=True
        )

        # Group states with same clean energy percentage
        result = []
        i = 0
        while i < len(sorted_states) and len(result) < bucket_size:
            current_percentage = sorted_states[i].get_clean_energy_percentage()

            # Find all states with same percentage
            same_percentage_states = []
            j = i
            while (
                j < len(sorted_states)
                and sorted_states[j].get_clean_energy_percentage() == current_percentage
            ):
                same_percentage_states.append(sorted_states[j])
                j += 1

            # Randomly select from ties
            available_slots = bucket_size - len(result)
            selected = random.sample(
                same_percentage_states,
                min(len(same_percentage_states), available_slots),
            )
            result.extend(selected)

            i = j

        return result

    def _bucket_success_small_miss(self, states: List[TreeNode]) -> List[TreeNode]:
        """Bucket: Success + Small Miss - most success + small miss combined, random tiebreak."""
        bucket_size = self.config["beam_search"]["bucket_sizes"]["success_small_miss"]

        # Sort by success + small miss count
        sorted_states = sorted(
            states,
            key=lambda x: x.get_decision_counts()["success"]
            + x.get_decision_counts()["small_miss"],
            reverse=True,
        )

        # Group states with same combined count
        result = []
        i = 0
        while i < len(sorted_states) and len(result) < bucket_size:
            current_combined = (
                sorted_states[i].get_decision_counts()["success"]
                + sorted_states[i].get_decision_counts()["small_miss"]
            )

            # Find all states with same combined count
            same_combined_states = []
            j = i
            while j < len(sorted_states):
                combined_count = (
                    sorted_states[j].get_decision_counts()["success"]
                    + sorted_states[j].get_decision_counts()["small_miss"]
                )
                if combined_count == current_combined:
                    same_combined_states.append(sorted_states[j])
                    j += 1
                else:
                    break

            # Randomly select from ties
            available_slots = bucket_size - len(result)
            selected = random.sample(
                same_combined_states,
                min(len(same_combined_states), available_slots),
            )
            result.extend(selected)

            i = j

        return result

    def _bucket_success(self, states: List[TreeNode]) -> List[TreeNode]:
        """Bucket: Success - most successes, random tiebreak."""
        bucket_size = self.config["beam_search"]["bucket_sizes"]["success"]

        # Sort by success count
        sorted_states = sorted(
            states, key=lambda x: x.get_decision_counts()["success"], reverse=True
        )

        # Group states with same success count
        result = []
        i = 0
        while i < len(sorted_states) and len(result) < bucket_size:
            current_success = sorted_states[i].get_decision_counts()["success"]

            # Find all states with same success count
            same_success_states = []
            j = i
            while (
                j < len(sorted_states)
                and sorted_states[j].get_decision_counts()["success"] == current_success
            ):
                same_success_states.append(sorted_states[j])
                j += 1

            # Randomly select from ties
            available_slots = bucket_size - len(result)
            selected = random.sample(
                same_success_states, min(len(same_success_states), available_slots)
            )
            result.extend(selected)

            i = j

        return result

    def _bucket_least_total_energy(self, states: List[TreeNode]) -> List[TreeNode]:
        """Bucket: Least Total Energy - lowest total energy consumption, random tiebreak."""
        bucket_size = self.config["beam_search"]["bucket_sizes"]["least_total_energy"]

        # Sort by total energy (ascending for least energy)
        sorted_states = sorted(
            states, key=lambda x: x.get_total_energy(), reverse=False
        )

        # Group states with same total energy
        result = []
        i = 0
        while i < len(sorted_states) and len(result) < bucket_size:
            current_energy = sorted_states[i].get_total_energy()

            # Find all states with same energy
            same_energy_states = []
            j = i
            while (
                j < len(sorted_states)
                and sorted_states[j].get_total_energy() == current_energy
            ):
                same_energy_states.append(sorted_states[j])
                j += 1

            # Randomly select from ties
            available_slots = bucket_size - len(result)
            selected = random.sample(
                same_energy_states,
                min(len(same_energy_states), available_slots),
            )
            result.extend(selected)

            i = j

        return result

    def _calculate_illogical_action_penalty(self, state: TreeNode) -> float:
        """Calculate penalty for illogical action sequences."""
        if not state.action_sequence:
            return 0.0

        penalty = 0.0
        battery_capacity = self.config["battery"]["capacity_wh"]

        for i, (model_name, should_charge) in enumerate(state.action_sequence):
            # Penalize IDLE when models were likely available
            if model_name == "IDLE":
                # Check if this was likely illogical (simplified heuristic)
                if i > 0:  # Not the first action
                    penalty += 2.0  # Significant penalty for unnecessary IDLE

            # Penalize charging when battery was nearly full
            if should_charge and state.parent:
                parent_battery = state.parent.battery_level_wh if state.parent else 0
                if parent_battery > battery_capacity * 0.95:  # >95% full
                    penalty += 0.5

        return penalty

    def _assign_to_buckets(self, states: List[TreeNode]) -> List[TreeNode]:
        """Assign states to Pareto buckets with minimum preservation logic."""
        if not states:
            return []

        # Filter valid states
        valid_states = []
        for state in states:
            if self._is_valid_state(state):
                valid_states.append(state)

        # Remove duplicates while preserving diversity
        unique_states = []
        seen_states = set()
        for state in valid_states:
            state_key = (
                round(state.battery_level_wh, 6),
                round(state.agg_clean_energy, 6),
                round(state.agg_dirty_energy, 6),
            )
            if state_key not in seen_states:
                seen_states.add(state_key)
                unique_states.append(state)

        # If we have fewer states than bucket capacity, keep all
        total_bucket_capacity = sum(self.config["beam_search"]["bucket_sizes"].values())
        if len(unique_states) <= total_bucket_capacity:
            return unique_states

        # Assign to buckets with diversity preservation
        bucket_results = []
        used_states = set()

        # Helper to get state key for exclusion
        def get_state_key(state: TreeNode) -> tuple:
            return (
                round(state.battery_level_wh, 6),
                round(state.agg_clean_energy, 6),
                round(state.agg_dirty_energy, 6),
            )

        # Each bucket picks from remaining states, excluding already selected ones
        remaining_states = unique_states.copy()

        success_bucket = self._bucket_success(remaining_states)
        bucket_results.extend(success_bucket)
        for state in success_bucket:
            used_states.add(get_state_key(state))

        remaining_states = [
            s for s in unique_states if get_state_key(s) not in used_states
        ]
        success_small_miss_bucket = self._bucket_success_small_miss(remaining_states)
        bucket_results.extend(success_small_miss_bucket)
        for state in success_small_miss_bucket:
            used_states.add(get_state_key(state))

        remaining_states = [
            s for s in unique_states if get_state_key(s) not in used_states
        ]
        clean_energy_bucket = self._bucket_most_clean_energy(remaining_states)
        bucket_results.extend(clean_energy_bucket)
        for state in clean_energy_bucket:
            used_states.add(get_state_key(state))

        remaining_states = [
            s for s in unique_states if get_state_key(s) not in used_states
        ]
        least_total_energy_bucket = self._bucket_least_total_energy(remaining_states)
        bucket_results.extend(least_total_energy_bucket)
        for state in least_total_energy_bucket:
            used_states.add(get_state_key(state))

        # Remove duplicates from merged buckets
        final_frontier = []
        seen_final = set()
        for state in bucket_results:
            state_key = (
                round(state.battery_level_wh, 6),
                round(state.agg_clean_energy, 6),
                round(state.agg_dirty_energy, 6),
            )
            if state_key not in seen_final:
                seen_final.add(state_key)
                final_frontier.append(state)

        return final_frontier

    def _is_valid_state(self, node: TreeNode) -> bool:
        """Check if state is valid for beam search with enhanced battery validation."""
        battery_capacity_wh = self.config["battery"]["capacity_wh"]

        # Battery bounds check
        if node.battery_level_wh < 0:
            return False
        if node.battery_level_wh > battery_capacity_wh:
            return False

        # Validate action sequence logic
        if node.action_sequence:
            last_action = node.action_sequence[-1]
            model_name, should_charge = last_action

            # IDLE should only be allowed when charging
            if model_name == "IDLE" and not should_charge:
                return False

            # Charging should only be allowed when battery < capacity
            if should_charge:
                # Check previous battery level to validate charging was allowed
                if node.parent and node.parent.battery_level_wh >= battery_capacity_wh:
                    return False

        return True

    def _validate_net_battery_change(
        self, current_battery: float, charge_energy: float, model_energy: float
    ) -> bool:
        """Validate that net battery change stays within bounds."""
        battery_capacity_wh = self.config["battery"]["capacity_wh"]
        net_battery = current_battery + charge_energy - model_energy
        return 0 <= net_battery <= battery_capacity_wh

    def _count_decision_outcomes(self, decision_history: List[Dict]) -> Dict[str, int]:
        """Count different decision outcomes."""
        counts = {"success": 0, "small_miss": 0, "large_miss": 0}
        for decision in decision_history:
            outcome = decision["outcome"]
            if outcome in counts:
                counts[outcome] += 1
        return counts

    def _serialize_leaf(self, leaf: TreeNode) -> Dict:
        """Serialize leaf node to dictionary with detailed action sequence."""
        decision_counts = self._count_decision_outcomes(leaf.decision_history)

        # Create detailed action sequence with state information
        detailed_action_sequence = []

        # Track running state
        current_battery = self.config["battery"][
            "capacity_wh"
        ]  # Start with full battery
        current_clean = 0.0
        current_dirty = 0.0

        # Process each action in sequence
        for i, decision in enumerate(leaf.decision_history):
            action = leaf.action_sequence[i]
            model_name, should_charge = action
            outcome = decision["outcome"]

            # Store state before action
            battery_before = current_battery
            clean_before = current_clean
            dirty_before = current_dirty

            # Get charge energy from decision history (stored during tree search)
            charge_energy = decision.get("charge_energy_used", 0.0)
            model_energy = 0.0

            if should_charge:
                # Debug logging for charging (using stored value)
                self._log_with_season(
                    f"Charging: stored={charge_energy:.8f}Wh", "debug"
                )

            if model_name != "IDLE":
                model_energy = (
                    self.models[model_name]["energy_per_inference_mwh"] / 1000.0
                )

            # Get clean energy percentage for this timestep
            timestamp = i * self.config["simulation"]["task_interval_seconds"]
            clean_pct = self.energy_data.get(int(timestamp % 86400), 50.0)

            # Calculate state after action (atomic update)
            current_battery = current_battery + charge_energy - model_energy

            # Only add clean/dirty energy when charging occurs
            if should_charge:
                current_clean = current_clean + (charge_energy * (clean_pct / 100.0))
                current_dirty = current_dirty + (
                    charge_energy * ((100 - clean_pct) / 100.0)
                )

            # Model energy does NOT add to dirty energy (only charging affects energy accounting)
            # Model usage only affects battery level, not clean/dirty energy tracking

            # Create detailed action record
            detailed_action = {
                "timestep": i,
                "model": model_name,
                "charged": should_charge,
                "outcome": outcome,
                "battery_before": battery_before,
                "battery_after": current_battery,
                "clean_energy_before": clean_before,
                "clean_energy_after": current_clean,
                "dirty_energy_before": dirty_before,
                "dirty_energy_after": current_dirty,
                "charge_energy": charge_energy,
                "model_energy": model_energy,
                "clean_energy_pct": clean_pct,
            }

            detailed_action_sequence.append(detailed_action)

        return {
            "action_sequence": detailed_action_sequence,
            "final_battery": leaf.battery_level_wh,
            "agg_clean_energy": round(leaf.agg_clean_energy, 6),
            "agg_dirty_energy": round(leaf.agg_dirty_energy, 6),
            "total_energy": round(leaf.agg_clean_energy + leaf.agg_dirty_energy, 6),
            "clean_energy_percentage": round(
                (
                    leaf.agg_clean_energy
                    / (leaf.agg_clean_energy + leaf.agg_dirty_energy)
                    * 100
                )
                if (leaf.agg_clean_energy + leaf.agg_dirty_energy) > 0
                else 0,
                2,
            ),
            "decision_counts": decision_counts,
            "total_decisions": len(leaf.decision_history),
        }

    def _categorize_results(self, leaves: List[TreeNode]) -> Dict:
        """Categorize results into top 5 for each category."""
        if not leaves:
            return {
                "top_success": [],
                "top_success_small_miss": [],
                "top_most_clean_energy": [],
                "top_least_total_energy": [],
            }

        # Sort by different metrics
        top_success = sorted(
            leaves,
            key=lambda x: self._count_decision_outcomes(x.decision_history)["success"],
            reverse=True,
        )[:10]

        top_success_small_miss = sorted(
            leaves,
            key=lambda x: (
                self._count_decision_outcomes(x.decision_history)["success"]
                + self._count_decision_outcomes(x.decision_history)["small_miss"]
            ),
            reverse=True,
        )[:10]

        top_clean_energy = sorted(
            leaves,
            key=lambda x: x.get_clean_energy_percentage(),
            reverse=True,
        )[:10]

        top_least_total_energy = sorted(
            leaves,
            key=lambda x: x.get_total_energy(),
            reverse=False,  # Ascending for least energy
        )[:10]

        return {
            "top_success": [self._serialize_leaf(leaf) for leaf in top_success],
            "top_success_small_miss": [
                self._serialize_leaf(leaf) for leaf in top_success_small_miss
            ],
            "top_most_clean_energy": [
                self._serialize_leaf(leaf) for leaf in top_clean_energy
            ],
            "top_least_total_energy": [
                self._serialize_leaf(leaf) for leaf in top_least_total_energy
            ],
        }

    def run_search(self) -> Dict:
        """Run complete tree search."""
        self._log_with_season(f"Starting tree search for {self.location} {self.season}")
        self._log_with_season(f"Horizon: {self.horizon} timesteps")
        self._log_with_season(f"Parallel mode: {self.parallel}")

        # Safety checks to prevent infinite loops
        max_safe_nodes = float("inf")  # 20M node limit
        max_safe_runtime = float("inf")  # 5 minute limit for workers
        max_safe_depth = float("inf")  # Prevent excessive depth

        # Initialize timing
        self.start_time = time.time()
        self.last_progress_time = self.start_time

        # Create root node
        root = TreeNode(
            battery_level_wh=self.config["battery"][
                "capacity_wh"
            ],  # Start fully charged
            timestep=0,
            action_sequence=[],
            agg_clean_energy=0.0,
            agg_dirty_energy=0.0,
            decision_history=[],
        )

        self._log_with_season(
            f"Safety limits: max_nodes={max_safe_nodes:,}, max_runtime={max_safe_runtime}s, max_depth={max_safe_depth}"
        )
        self._log_with_season(
            f"Battery: {self.config['battery']['capacity_wh']}Wh, Charge: {self.config['battery']['charge_rate_hours']}h"
        )

        # Expand tree (always parallel hybrid + naive search)
        self._log_with_season("Expanding decision tree and running naive search...")

        # Force parallel mode for tree search
        original_parallel = self.parallel
        self.parallel = True  # Always use parallel

        # Start naive search worker in parallel
        naive_start_time = time.time()
        with ProcessPoolExecutor(max_workers=2) as executor:
            # Submit tree search (always parallel hybrid)
            tree_future = executor.submit(self._expand_hybrid_parallel, root)

            # Submit naive search
            naive_future = executor.submit(
                self._run_naive_search_worker,
                (self.config_path, self.location, self.season),
            )

            # Collect results
            try:
                leaves = tree_future.result()  # No timeout
            except Exception as e:
                self._log_with_season(f"Tree search failed: {e}", "error")
                leaves = []

            try:
                naive_result = naive_future.result()  # No timeout
                naive_runtime = time.time() - naive_start_time
                self._log_with_season(
                    f"Naive search completed in {naive_runtime:.2f} seconds"
                )
            except Exception as e:
                self._log_with_season(f"Naive search failed: {e}", "error")
                naive_result = None
                naive_runtime = 0

        # Restore original parallel setting for metadata
        self.parallel = original_parallel

        total_time = time.time() - self.start_time
        self._log_with_season(f"Tree expansion complete in {total_time:.1f} seconds:")
        self._log_with_season(f"  Nodes explored: {self.nodes_explored:,}")
        self._log_with_season(f"  Nodes pruned: {self.nodes_pruned:,}")
        self._log_with_season(f"  Children found: {self.children_found:,}")
        self._log_with_season(f"  Children pruned: {self.children_pruned:,}")
        self._log_with_season(f"  Leaf nodes: {len(leaves):,}")
        self._log_with_season(f"  Leaves found: {self.leaves_found:,}")
        if self.nodes_explored + self.nodes_pruned > 0:
            self._log_with_season(
                f"  Pruning efficiency: {self.nodes_pruned / (self.nodes_explored + self.nodes_pruned) * 100:.1f}%"
            )
        else:
            self._log_with_season("  Pruning efficiency: N/A (no nodes processed)")
        self._log_with_season(
            f"  Average rate: {self.nodes_explored / total_time:.0f} nodes/sec"
        )
        if hasattr(self, "cache_hits"):
            self._log_with_season(f"  Cache hits: {self.cache_hits:,}")

        # Categorize results
        self._log_with_season("Categorizing results...")
        results = self._categorize_results(leaves)

        # Add naive results if available
        if naive_result:
            naive_serialized = self._serialize_leaf(naive_result)
            results["naive"] = naive_serialized
            self._log_with_season("Added naive search results to output")
        else:
            self._log_with_season("No naive results available", "warning")

        # Create final output
        output = {
            "metadata": {
                "location": self.location,
                "horizon": self.horizon,
                "total_leaves_explored": len(leaves),
                "runtime_seconds": round(total_time, 2),
                "naive_runtime_seconds": round(naive_runtime, 2) if naive_result else 0,
                "timestamp": datetime.now().isoformat(),
                "safety_limits_hit": self.nodes_explored >= max_safe_nodes
                or total_time >= max_safe_runtime,
                "parallel_mode": self.parallel,
                "energy_data_location": self.location,
                "energy_data_date": getattr(self, "energy_data_date", ""),
                "user_accuracy_requirement": self.config["simulation"][
                    "user_accuracy_requirement"
                ],
                "user_latency_requirement": self.config["simulation"][
                    "user_latency_requirement"
                ],
                "duration_days": self.config["simulation"]["duration_days"],
                "task_interval_seconds": self.config["simulation"][
                    "task_interval_seconds"
                ],
                "battery_capacity_wh": self.config["battery"]["capacity_wh"],
                "charge_rate_hours": self.config["battery"]["charge_rate_hours"],
            },
            "results": results,
        }

        return output


def _run_season_worker(
    config_path: str, location: str, season: str
) -> Tuple[str, Dict]:
    """Worker function for running tree search for a single season."""
    tree_search = TreeSearch(config_path, location, season, parallel=True)
    results = tree_search.run_search()
    return season, results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Tree search for battery-powered model selection"
    )
    parser.add_argument(
        "--location",
        default="CA",
        choices=["CA", "FL", "NW", "NY"],
        help="Geographic location for energy data",
    )
    parser.add_argument(
        "--config", default="config.jsonc", help="Configuration file path"
    )
    parser.add_argument(
        "--output", help="Output file path (ignored - timestamp-based naming used)"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable hybrid parallel processing (sequential expansion + parallel beam search)",
    )

    args = parser.parse_args()

    # Define all seasons to run
    seasons = ["winter", "spring", "summer", "fall"]

    # Generate timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Starting tree search for all 4 seasons in parallel...")
    print(f"Location: {args.location}")
    print(f"Timestamp: {timestamp}")

    # Run all seasons in parallel with 4 workers (1 per season)
    all_results = {}
    with ProcessPoolExecutor(max_workers=4) as executor:
        # Submit all seasons as separate tasks
        future_to_season = {
            executor.submit(
                _run_season_worker, args.config, args.location, season
            ): season
            for season in seasons
        }

        # Collect results as they complete
        for future in as_completed(future_to_season):  # No timeout
            season = future_to_season[future]
            try:
                season_result = future.result()
                season, results = season_result
                all_results[season] = results
                print(f"[{season.upper()}] Completed successfully!")
            except Exception as e:
                print(f"[{season.upper()}] Failed: {e}")
                # Cancel remaining futures and exit
                for remaining_future in future_to_season:
                    remaining_future.cancel()
                print("Tree search failed - exiting without saving results")
                return 1

    # Save results for each season with season-specific filename
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    for season, results in all_results.items():
        output_filename = f"{args.location}-{timestamp}-{season}-metadata.json"
        output_path = results_dir / output_filename

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"[{season.upper()}] Results saved to {output_path}")
        print(
            f"[{season.upper()}] Explored {results['metadata']['total_leaves_explored']} leaf nodes"
        )

    print("All seasons completed successfully!")
    return 0


if __name__ == "__main__":
    main()
