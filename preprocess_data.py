#!/usr/bin/env python3
"""
Fast data preprocessing script to extract training data from beam search results.
Processes all JSON files in results/ and creates training datasets for each bucket.
Minimal output, optimized for large datasets.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional


def load_metadata_file(filepath: str) -> Dict[str, Any]:
    """Load and parse metadata JSON file."""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def validate_timestep_data(timestep: Dict[str, Any]) -> bool:
    """Validate that timestep contains required fields for training."""
    required_fields = [
        "battery_before",
        "clean_energy_pct",
        "model",
        "charged",
        "outcome",
    ]
    return all(field in timestep for field in required_fields)


def validate_metadata(metadata: Dict[str, Any]) -> bool:
    """Validate that metadata contains required configuration fields."""
    required_fields = [
        "battery_capacity_wh",
        "charge_rate_hours",
        "task_interval_seconds",
        "user_accuracy_requirement",
        "user_latency_requirement",
    ]
    return all(field in metadata for field in required_fields)


def extract_bucket_sequence(
    data: Dict[str, Any], bucket_key: str
) -> List[Dict[str, Any]]:
    """Extract the optimal action sequence from a specific bucket."""
    try:
        bucket_data = data.get("results", {}).get(bucket_key, [])
        if bucket_data and len(bucket_data) > 0:
            sequence = bucket_data[0].get("action_sequence", [])
            return [
                timestep for timestep in sequence if validate_timestep_data(timestep)
            ]
        return []
    except Exception:
        return []


def process_timestep_to_training_sample(
    timestep: Dict[str, Any], metadata: Dict[str, Any], source_file: str
) -> Optional[Dict[str, Any]]:
    """Convert a single timestep to training sample format."""
    try:
        battery_level = float(timestep.get("battery_before", 0.0))
        clean_energy_pct = float(timestep.get("clean_energy_pct", 0.0))
        battery_capacity_wh = float(metadata.get("battery_capacity_wh", 0.0))
        charge_rate_hours = float(metadata.get("charge_rate_hours", 0.0))
        task_interval_seconds = int(metadata.get("task_interval_seconds", 300))
        user_accuracy_req = float(metadata.get("user_accuracy_requirement", 0.905))
        user_latency_req = float(metadata.get("user_latency_requirement", 0.01))
        optimal_model = str(timestep.get("model", ""))
        should_charge = bool(timestep.get("charged", False))
        outcome = str(timestep.get("outcome", ""))

        battery_percentage = (
            (battery_level / battery_capacity_wh) * 100
            if battery_capacity_wh > 0
            else 0.0
        )
        energy_efficiency_score = clean_energy_pct * (1 - battery_percentage / 100)

        return {
            "battery_level": battery_level,
            "battery_percentage": battery_percentage,
            "clean_energy_percentage": clean_energy_pct,
            "battery_capacity_wh": battery_capacity_wh,
            "charge_rate_hours": charge_rate_hours,
            "task_interval_seconds": task_interval_seconds,
            "user_accuracy_requirement": user_accuracy_req,
            "user_latency_requirement": user_latency_req,
            "energy_efficiency_score": energy_efficiency_score,
            "time_to_full_charge": (battery_capacity_wh - battery_level)
            / (battery_capacity_wh / charge_rate_hours)
            if charge_rate_hours > 0
            else 0.0,
            "optimal_model": optimal_model,
            "should_charge": should_charge,
            "outcome": outcome,
            "source_file": source_file,
        }
    except Exception:
        return None


def process_single_file(filepath: str) -> Dict[str, List[Dict[str, Any]]]:
    """Process a single metadata JSON file and extract bucket sequences."""
    data = load_metadata_file(filepath)
    if not data:
        return {
            "success": [],
            "success_small_miss": [],
            "most_clean_energy": [],
            "least_total_energy": [],
        }

    metadata = data.get("metadata", {})
    if not validate_metadata(metadata):
        return {
            "success": [],
            "success_small_miss": [],
            "most_clean_energy": [],
            "least_total_energy": [],
        }

    bucket_mapping = {
        "success": "top_success",
        "success_small_miss": "top_success_small_miss",
        "most_clean_energy": "top_most_clean_energy",
        "least_total_energy": "top_least_total_energy",
    }

    buckets = {}
    for bucket_name, bucket_key in bucket_mapping.items():
        buckets[bucket_name] = extract_bucket_sequence(data, bucket_key)

    processed_buckets = {}
    for bucket_name, sequence in buckets.items():
        training_samples = []
        for timestep in sequence:
            sample = process_timestep_to_training_sample(
                timestep, metadata, Path(filepath).name
            )
            if sample:
                training_samples.append(sample)
        processed_buckets[bucket_name] = training_samples

    return processed_buckets


def save_training_data(
    bucket_samples: Dict[str, List[Dict[str, Any]]], output_dir: str
) -> None:
    """Save training data to JSON files."""
    os.makedirs(output_dir, exist_ok=True)

    bucket_filenames = {
        "success": "success-training-data.json",
        "success_small_miss": "success_small_miss-training-data.json",
        "most_clean_energy": "most_clean_energy-training-data.json",
        "least_total_energy": "least_total_energy-training-data.json",
    }

    for bucket_name, samples in bucket_samples.items():
        filename = bucket_filenames[bucket_name]
        filepath = os.path.join(output_dir, filename)

        dataset_info = {
            "bucket_name": bucket_name,
            "total_samples": len(samples),
            "training_samples": samples,
        }

        with open(filepath, "w") as f:
            json.dump(dataset_info, f, separators=(",", ":"))


def main():
    """Main preprocessing pipeline."""
    results_dir = Path("results")
    output_dir = Path("training-data")

    if not results_dir.exists():
        return

    json_files = list(results_dir.glob("*-metadata.json"))
    if not json_files:
        return

    all_bucket_data = {
        "success": [],
        "success_small_miss": [],
        "most_clean_energy": [],
        "least_total_energy": [],
    }

    for filepath in sorted(json_files):
        file_buckets = process_single_file(str(filepath))
        for bucket_name, samples in file_buckets.items():
            all_bucket_data[bucket_name].extend(samples)

    save_training_data(all_bucket_data, str(output_dir))


if __name__ == "__main__":
    main()
