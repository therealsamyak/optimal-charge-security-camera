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
import multiprocessing as mp
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
        self.energy_data = self._load_energy_data()

        # Calculate search parameters
        self.horizon = int(
            (self.config["simulation"]["duration_days"] * 24 * 3600)
            / self.config["simulation"]["task_interval_seconds"]
        )

        # Setup logging
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
                "most_clean_energy_usage": 10,
                "h_1": 10,
                "h_2": 10,
                "h_3": 10,
                "h_4": 10,
            },
        )

        # Beam search performance tracking
        self.frontier_sizes = []
        self.bucket_utilization = {bucket: [] for bucket in self.bucket_sizes.keys()}

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

    def _load_energy_data(self) -> Dict[int, float]:
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

        # Filter by season and interpolate
        season_data = self._filter_by_season(energy_data)
        return self._interpolate_energy_data(season_data)

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

    def _get_valid_actions(self, node: TreeNode) -> List[Tuple[str, bool]]:
        """Get valid actions for current node - generate ALL possible model actions."""
        actions = []
        battery_level_wh = node.battery_level_wh
        battery_capacity_wh = self.config["battery"]["capacity_wh"]

        # Pruning: battery = 0% → only idle+charging
        if battery_level_wh <= 0:
            return [("IDLE", True)]

        # Pruning: battery = 100% → no charging actions
        can_charge = battery_level_wh < battery_capacity_wh

        # Get all models that can run with current battery level
        runnable_models = []
        for model_name, model_spec in self.models.items():
            energy_needed_wh = model_spec["energy_per_inference_mwh"] / 1000.0
            if battery_level_wh >= energy_needed_wh:
                runnable_models.append(model_name)

        # Log valid models at first timestep for debugging
        if node.timestep == 0 and not hasattr(self, "_logged_models"):
            task_req = {
                "accuracy": self.config["simulation"]["user_accuracy_requirement"]
                / 100.0,
                "latency": self.config["simulation"]["user_latency_requirement"],
            }
            qualified_models = [
                name
                for name in runnable_models
                if (
                    self.models[name]["accuracy"] >= task_req["accuracy"]
                    and self.models[name]["latency"] <= task_req["latency"]
                )
            ]
            fallback_models = [
                name for name in runnable_models if name not in qualified_models
            ]
            self.logger.info(
                f"Models meeting requirements (accuracy>={task_req['accuracy']:.2f}, latency<={task_req['latency']:.3f}s): {qualified_models}"
            )
            self.logger.info(f"Fallback models available: {fallback_models}")
            self.logger.info(
                f"Battery capacity: {self.config['battery']['capacity_wh']}Wh, Charge rate: {self.config['battery']['charge_rate_hours']}h"
            )
            self._logged_models = True

        # Generate ALL possible actions for runnable models
        for model_name in runnable_models:
            actions.append((model_name, False))  # Without charging
            if can_charge:
                actions.append((model_name, True))  # With charging

        # Add IDLE+charging ONLY if no models can run
        if not runnable_models and can_charge:
            actions.append(("IDLE", True))

        return actions

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
            self.logger.debug(
                f"Charging: raw={charge_energy_wh:.8f}Wh, space={space_available:.8f}Wh, actual={actual_charge:.8f}Wh"
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
            # Model energy already accounted for in net calculation
            new_dirty_energy += model_energy_wh  # Model usage is dirty energy

            # Check if chosen model meets user requirements
            task_req = {
                "accuracy": self.config["simulation"]["user_accuracy_requirement"]
                / 100.0,
                "latency": self.config["simulation"]["user_latency_requirement"],
            }

            model_spec = self.models[model_name]
            meets_accuracy = model_spec["accuracy"] >= task_req["accuracy"]
            meets_latency = model_spec["latency"] <= task_req["latency"]

            if meets_accuracy and meets_latency:
                decision_outcome = "success"
            else:
                # Chosen model doesn't meet requirements = small miss
                decision_outcome = "small_miss"

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

    def _expand_subtree_worker(self, worker_args: Tuple) -> List[TreeNode]:
        """Worker function for parallel subtree expansion."""
        node, depth, config_path, location, season = worker_args

        # Create worker-specific TreeSearch instance
        worker_search = TreeSearch(config_path, location, season, parallel=False)
        worker_search.nodes_explored = 0
        worker_search.nodes_pruned = 0
        worker_search.leaves_found = 0
        worker_search.children_found = 0
        worker_search.children_pruned = 0

        return worker_search._expand_tree_fallback(node, depth)

    def _expand_tree_parallel(
        self, root: TreeNode, split_depths: List[int] = [5, 10, 15]
    ) -> List[TreeNode]:
        """Expand tree using parallel depth-based splitting."""
        all_leaves = []
        max_workers = min(self.config["workers"]["max_workers"], mp.cpu_count())

        self.logger.info(f"Starting parallel expansion with {max_workers} workers")

        # Expand to first split depth
        current_nodes = [root]
        current_depth = 0

        for split_depth in split_depths:
            if current_depth >= split_depth or current_depth >= self.horizon:
                break

            self.logger.info(f"Expanding to depth {split_depth}...")
            next_nodes = []

            for node in current_nodes:
                # Get clean energy percentage for current timestep
                timestamp = (
                    node.timestep * self.config["simulation"]["task_interval_seconds"]
                )
                clean_pct = self.energy_data.get(int(timestamp % 86400), 50.0)

                valid_actions = self._get_valid_actions(node)
                for action in valid_actions:
                    child = self._apply_action(node, action, clean_pct)
                    if child:
                        next_nodes.append(child)

            current_nodes = next_nodes
            current_depth = split_depth

            if len(current_nodes) >= max_workers * 10:  # Enough nodes to parallelize
                break

        # If we have enough nodes, parallelize the remaining expansion
        if len(current_nodes) >= max_workers and current_depth < self.horizon:
            self.logger.info(
                f"Parallelizing {len(current_nodes)} subtrees at depth {current_depth}"
            )

            # Prepare worker arguments
            worker_args = [
                (node, current_depth, self.config_path, self.location, self.season)
                for node in current_nodes
            ]

            # Process subtrees in parallel
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_node = {
                    executor.submit(self._expand_subtree_worker, args): args[0]
                    for args in worker_args
                }

                for future in as_completed(future_to_node):
                    try:
                        subtree_leaves = future.result(
                            timeout=300
                        )  # 5min timeout per worker
                        all_leaves.extend(subtree_leaves)
                        self.leaves_found += len(subtree_leaves)
                    except Exception as e:
                        self.logger.error(f"Worker failed: {e}")
                        # Fall back to sequential processing for this subtree
                        node = future_to_node[future]
                        try:
                            sequential_leaves = self._expand_tree_fallback(
                                node, current_depth
                            )
                            all_leaves.extend(sequential_leaves)
                            self.leaves_found += len(sequential_leaves)
                        except Exception as e2:
                            self.logger.error(f"Sequential fallback also failed: {e2}")
        else:
            # Not enough nodes to parallelize, use sequential
            self.logger.info(
                "Using sequential expansion (insufficient nodes for parallelization)"
            )
            all_leaves = self._expand_tree_fallback(root, 0)

        return all_leaves

    def _beam_search_expand(self, root: TreeNode) -> List[TreeNode]:
        """Pareto bucketed beam search for efficient tree exploration."""
        if not self.config.get("beam_search", {}).get("enabled", False):
            # Fall back to original tree search if beam search disabled
            return self._expand_tree_fallback(root, 0)

        self.logger.info("Starting Pareto bucketed beam search")

        # Initialize frontier with root node
        frontier = [root]
        max_frontier_size = self.config["beam_search"]["max_frontier_size"]

        # Track beam search statistics
        beam_stats = {
            "avg_frontier_size": 0,
            "total_states_generated": 0,
            "total_states_pruned": 0,
            "bucket_utilization": {
                "success": 0,
                "most_clean_energy_usage": 0,
                "h_1": 0,
                "h_2": 0,
                "h_3": 0,
                "h_4": 0,
            },
        }

        for timestep in range(self.horizon):
            if not frontier:
                self.logger.error(f"All buckets empty at timestep {timestep}")
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

            # Assign states to buckets and prune
            frontier = self._assign_to_buckets(next_states)
            beam_stats["total_states_pruned"] += len(next_states) - len(frontier)

            # Progress logging
            if timestep % 50 == 0 or timestep == self.horizon - 1:
                self.logger.info(
                    f"Timestep {timestep}/{self.horizon}: "
                    f"frontier_size={len(frontier)}, "
                    f"states_generated={len(next_states)}, "
                    f"states_pruned={len(next_states) - len(frontier)}"
                )

            # Safety check for frontier explosion
            if len(frontier) > max_frontier_size * 2:
                self.logger.warning(
                    f"Frontier size exceeded safety limit: {len(frontier)}"
                )
                # Emergency pruning - keep only top states by clean energy
                frontier.sort(
                    key=lambda x: x.get_clean_energy_percentage(), reverse=True
                )
                frontier = frontier[:max_frontier_size]

        # Final statistics
        if self.horizon > 0:
            beam_stats["avg_frontier_size"] /= self.horizon

        self.logger.info("Beam search complete:")
        self.logger.info(
            f"  Average frontier size: {beam_stats['avg_frontier_size']:.1f}"
        )
        self.logger.info(
            f"  Total states generated: {beam_stats['total_states_generated']:,}"
        )
        self.logger.info(
            f"  Total states pruned: {beam_stats['total_states_pruned']:,}"
        )
        self.logger.info(f"  Final frontier size: {len(frontier)}")

        # Store beam search metadata
        self.beam_stats = beam_stats

        return frontier

    def _expand_tree_fallback(self, node: TreeNode, depth: int) -> List[TreeNode]:
        """Fallback to original tree expansion for compatibility."""
        # Safety checks to prevent infinite loops
        max_safe_nodes = 1000000  # 10M node limit
        max_safe_runtime = 300  # 5 minute limit
        max_safe_depth = 1000  # Prevent excessive depth

        # Use BFS with explicit stack to avoid recursion
        stack = [(node, depth)]
        all_leaves = []

        while stack:
            current_node, current_depth = stack.pop()
            self.nodes_explored += 1

            # Safety checks
            if self.nodes_explored >= max_safe_nodes:
                self.logger.warning(f"SAFETY: Node limit reached ({max_safe_nodes:,})")
                break
            if current_depth >= max_safe_depth:
                self.logger.warning(f"SAFETY: Depth limit reached ({max_safe_depth:,})")
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

                    self.logger.info(
                        f"Progress: {self.nodes_explored:,} nodes explored, "
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
                self.logger.warning(
                    f"SAFETY: Runtime limit reached ({max_safe_runtime}s)"
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
                self.logger.warning(
                    "SAFETY: Tree expansion terminated due to safety limits"
                )
                break

        return all_leaves

        # Safety checks to prevent infinite loops
        max_safe_nodes = 1000000  # 10M node limit
        max_safe_runtime = 300  # 5 minute limit
        max_safe_depth = 1000  # Prevent excessive depth

        # Use BFS with explicit stack to avoid recursion
        stack = [(node, depth)]
        all_leaves = []

        while stack:
            current_node, current_depth = stack.pop()
            self.nodes_explored += 1

            # Safety checks
            if self.nodes_explored >= max_safe_nodes:
                self.logger.warning(f"SAFETY: Node limit reached ({max_safe_nodes:,})")
                break
            if current_depth >= max_safe_depth:
                self.logger.warning(f"SAFETY: Depth limit reached ({max_safe_depth:,})")
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

                    self.logger.info(
                        f"Progress: {self.nodes_explored:,} nodes explored, "
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
                self.logger.warning(
                    f"SAFETY: Runtime limit reached ({max_safe_runtime}s)"
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
                self.logger.warning(
                    "SAFETY: Tree expansion terminated due to safety limits"
                )
                break

    def _bucket_most_clean_energy_usage(self, states: List[TreeNode]) -> List[TreeNode]:
        """Bucket: Most Clean Energy Usage - highest clean energy percentage, random tiebreak."""
        bucket_size = self.config["beam_search"]["bucket_sizes"][
            "most_clean_energy_usage"
        ]

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

    def _bucket_success_dominant(self, states: List[TreeNode]) -> List[TreeNode]:
        """Bucket: Success Dominant - most successes, random tiebreak."""
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

    def _calculate_heuristic_score(
        self, state: TreeNode, weights: Dict[str, float]
    ) -> float:
        """Calculate heuristic score for a state using given weights."""
        decision_counts = state.get_decision_counts()
        clean_energy_pct = state.get_clean_energy_percentage()

        # Normalize counts to 0-1 range (assuming max possible is horizon)
        max_possible = self.horizon
        success_score = (
            decision_counts["success"] / max_possible if max_possible > 0 else 0
        )
        small_miss_score = (
            decision_counts["small_miss"] / max_possible if max_possible > 0 else 0
        )
        large_miss_score = (
            decision_counts["large_miss"] / max_possible if max_possible > 0 else 0
        )
        clean_energy_score = clean_energy_pct / 100.0

        # Apply weights
        score = (
            weights["success"] * success_score
            + weights["small_miss"] * small_miss_score
            + weights["large_miss"] * large_miss_score
            + weights["clean_energy"] * clean_energy_score
        )

        return score

    def _bucket_heuristic(
        self, states: List[TreeNode], heuristic_name: str
    ) -> List[TreeNode]:
        """Generic heuristic bucket method with random tiebreak."""
        bucket_size = self.config["beam_search"]["bucket_sizes"][heuristic_name]
        weights = self.config["beam_search"]["heuristic_weights"][heuristic_name]

        # Calculate scores for all states
        scored_states = []
        for state in states:
            score = self._calculate_heuristic_score(state, weights)
            scored_states.append((score, state))

        # Sort by score (descending)
        scored_states.sort(key=lambda x: x[0], reverse=True)

        # Group states with same score and randomly select from ties
        result = []
        i = 0
        while i < len(scored_states) and len(result) < bucket_size:
            current_score = scored_states[i][0]

            # Find all states with same score
            same_score_states = []
            j = i
            while (
                j < len(scored_states)
                and abs(scored_states[j][0] - current_score) < 1e-10
            ):
                same_score_states.append(scored_states[j][1])
                j += 1

            # Randomly select from ties
            available_slots = bucket_size - len(result)
            selected = random.sample(
                same_score_states, min(len(same_score_states), available_slots)
            )
            result.extend(selected)

            i = j

        return result

    def _bucket_h_1(self, states: List[TreeNode]) -> List[TreeNode]:
        """Bucket: Heuristic 1 - balanced weights with success emphasis."""
        return self._bucket_heuristic(states, "h_1")

    def _bucket_h_2(self, states: List[TreeNode]) -> List[TreeNode]:
        """Bucket: Heuristic 2 - balanced weights with clean energy emphasis."""
        return self._bucket_heuristic(states, "h_2")

    def _bucket_h_3(self, states: List[TreeNode]) -> List[TreeNode]:
        """Bucket: Heuristic 3 - extreme success emphasis (0.8 weight)."""
        return self._bucket_heuristic(states, "h_3")

    def _bucket_h_4(self, states: List[TreeNode]) -> List[TreeNode]:
        """Bucket: Heuristic 4 - extreme clean energy emphasis (0.8 weight)."""
        return self._bucket_heuristic(states, "h_4")

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

        success_bucket = self._bucket_success_dominant(remaining_states)
        bucket_results.extend(success_bucket)
        for state in success_bucket:
            used_states.add(get_state_key(state))

        remaining_states = [
            s for s in unique_states if get_state_key(s) not in used_states
        ]
        clean_energy_bucket = self._bucket_most_clean_energy_usage(remaining_states)
        bucket_results.extend(clean_energy_bucket)
        for state in clean_energy_bucket:
            used_states.add(get_state_key(state))

        remaining_states = [
            s for s in unique_states if get_state_key(s) not in used_states
        ]
        h_1_bucket = self._bucket_h_1(remaining_states)
        bucket_results.extend(h_1_bucket)
        for state in h_1_bucket:
            used_states.add(get_state_key(state))

        remaining_states = [
            s for s in unique_states if get_state_key(s) not in used_states
        ]
        h_2_bucket = self._bucket_h_2(remaining_states)
        bucket_results.extend(h_2_bucket)
        for state in h_2_bucket:
            used_states.add(get_state_key(state))

        remaining_states = [
            s for s in unique_states if get_state_key(s) not in used_states
        ]
        h_3_bucket = self._bucket_h_3(remaining_states)
        bucket_results.extend(h_3_bucket)
        for state in h_3_bucket:
            used_states.add(get_state_key(state))

        remaining_states = [
            s for s in unique_states if get_state_key(s) not in used_states
        ]
        h_4_bucket = self._bucket_h_4(remaining_states)
        bucket_results.extend(h_4_bucket)

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
                self.logger.debug(f"Charging: stored={charge_energy:.8f}Wh")

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

            # Model energy is always dirty energy
            if model_name != "IDLE":
                current_dirty = current_dirty + model_energy

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
                "top_most_clean_energy_usage": [],
                "top_h_1": [],
                "top_h_2": [],
                "top_h_3": [],
                "top_h_4": [],
            }

        # Sort by different metrics
        top_success = sorted(
            leaves,
            key=lambda x: self._count_decision_outcomes(x.decision_history)["success"],
            reverse=True,
        )[:5]

        top_clean_energy = sorted(
            leaves,
            key=lambda x: x.get_clean_energy_percentage(),
            reverse=True,
        )[:5]

        # Get heuristic weights from config
        h_1_weights = self.config["beam_search"]["heuristic_weights"]["h_1"]
        h_2_weights = self.config["beam_search"]["heuristic_weights"]["h_2"]
        h_3_weights = self.config["beam_search"]["heuristic_weights"]["h_3"]
        h_4_weights = self.config["beam_search"]["heuristic_weights"]["h_4"]

        # Sort by heuristic scores
        top_h_1 = sorted(
            leaves,
            key=lambda x: self._calculate_heuristic_score(x, h_1_weights),
            reverse=True,
        )[:5]

        top_h_2 = sorted(
            leaves,
            key=lambda x: self._calculate_heuristic_score(x, h_2_weights),
            reverse=True,
        )[:5]

        top_h_3 = sorted(
            leaves,
            key=lambda x: self._calculate_heuristic_score(x, h_3_weights),
            reverse=True,
        )[:5]

        top_h_4 = sorted(
            leaves,
            key=lambda x: self._calculate_heuristic_score(x, h_4_weights),
            reverse=True,
        )[:5]

        return {
            "top_success": [self._serialize_leaf(leaf) for leaf in top_success],
            "top_most_clean_energy_usage": [
                self._serialize_leaf(leaf) for leaf in top_clean_energy
            ],
            "top_h_1": [self._serialize_leaf(leaf) for leaf in top_h_1],
            "top_h_2": [self._serialize_leaf(leaf) for leaf in top_h_2],
            "top_h_3": [self._serialize_leaf(leaf) for leaf in top_h_3],
            "top_h_4": [self._serialize_leaf(leaf) for leaf in top_h_4],
        }

    def run_search(self) -> Dict:
        """Run complete tree search."""
        self.logger.info(f"Starting tree search for {self.location} {self.season}")
        self.logger.info(f"Horizon: {self.horizon} timesteps")
        self.logger.info(f"Parallel mode: {self.parallel}")

        # Safety checks to prevent infinite loops
        max_safe_nodes = 20000000  # 20M node limit
        max_safe_runtime = 300  # 5 minute limit for workers
        max_safe_depth = 1000  # Prevent excessive depth

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

        self.logger.info(
            f"Safety limits: max_nodes={max_safe_nodes:,}, max_runtime={max_safe_runtime}s, max_depth={max_safe_depth}"
        )
        self.logger.info(
            f"Battery: {self.config['battery']['capacity_wh']}Wh, Charge: {self.config['battery']['charge_rate_hours']}h"
        )

        # Expand tree (beam search, parallel, or sequential)
        self.logger.info("Expanding decision tree...")
        if self.config.get("beam_search", {}).get("enabled", False):
            leaves = self._beam_search_expand(root)
        elif self.parallel:
            leaves = self._expand_tree_parallel(root)
        else:
            leaves = self._expand_tree_fallback(root, 0)

        total_time = time.time() - self.start_time
        self.logger.info(f"Tree expansion complete in {total_time:.1f} seconds:")
        self.logger.info(f"  Nodes explored: {self.nodes_explored:,}")
        self.logger.info(f"  Nodes pruned: {self.nodes_pruned:,}")
        self.logger.info(f"  Children found: {self.children_found:,}")
        self.logger.info(f"  Children pruned: {self.children_pruned:,}")
        self.logger.info(f"  Leaf nodes: {len(leaves):,}")
        self.logger.info(f"  Leaves found: {self.leaves_found:,}")
        if self.nodes_explored + self.nodes_pruned > 0:
            self.logger.info(
                f"  Pruning efficiency: {self.nodes_pruned / (self.nodes_explored + self.nodes_pruned) * 100:.1f}%"
            )
        else:
            self.logger.info("  Pruning efficiency: N/A (no nodes processed)")
        self.logger.info(
            f"  Average rate: {self.nodes_explored / total_time:.0f} nodes/sec"
        )
        if hasattr(self, "cache_hits"):
            self.logger.info(f"  Cache hits: {self.cache_hits:,}")

        # Categorize results
        self.logger.info("Categorizing results...")
        results = self._categorize_results(leaves)

        # Create final output
        output = {
            "metadata": {
                "location": self.location,
                "season": self.season,
                "horizon": self.horizon,
                "total_leaves_explored": len(leaves),
                "nodes_explored": self.nodes_explored,
                "nodes_pruned": self.nodes_pruned,
                "children_found": self.children_found,
                "children_pruned": self.children_pruned,
                "leaves_found": self.leaves_found,
                "pruning_efficiency": round(
                    self.nodes_pruned / (self.nodes_explored + self.nodes_pruned) * 100
                    if (self.nodes_explored + self.nodes_pruned) > 0
                    else 0,
                    2,
                ),
                "runtime_seconds": round(total_time, 2),
                "timestamp": datetime.now().isoformat(),
                "safety_limits_hit": self.nodes_explored >= max_safe_nodes
                or total_time >= max_safe_runtime,
                "parallel_mode": self.parallel,
                "cache_hits": getattr(self, "cache_hits", 0),
                "beam_search_enabled": self.config.get("beam_search", {}).get(
                    "enabled", False
                ),
                "beam_stats": getattr(self, "beam_stats", {}),
            },
            "results": results,
        }

        return output


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
        "--season",
        default="winter",
        choices=["winter", "spring", "summer", "fall"],
        help="Season for energy data",
    )
    parser.add_argument(
        "--config", default="config.jsonc", help="Configuration file path"
    )
    parser.add_argument(
        "--output", help="Output file path (ignored - timestamp-based naming used)"
    )
    parser.add_argument(
        "--parallel", action="store_true", help="Enable parallel processing"
    )

    args = parser.parse_args()

    # Run tree search
    tree_search = TreeSearch(args.config, args.location, args.season, args.parallel)
    results = tree_search.run_search()

    # Save results with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"tree-search-{timestamp}-metadata.json"
    output_path = Path("results") / output_filename
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Tree search complete! Results saved to {output_path}")
    print(f"Explored {results['metadata']['total_leaves_explored']} leaf nodes")
    print(f"Pruning efficiency: {results['metadata']['pruning_efficiency']}%")


if __name__ == "__main__":
    main()
