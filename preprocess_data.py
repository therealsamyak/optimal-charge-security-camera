#!/usr/bin/env python3
"""
Comprehensive data preprocessing script to extract training data from beam search results.
Processes all JSON files in results/ and creates balanced training datasets for each bucket.
Includes robust error handling, data validation, and detailed analytics for ML pipeline integration.
"""

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import argparse


def load_metadata_file(filepath: str) -> Dict[str, Any]:
    """Load and parse metadata JSON file with robust error handling."""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
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
            # Filter out invalid timesteps
            return [
                timestep for timestep in sequence if validate_timestep_data(timestep)
            ]
        return []
    except Exception as e:
        print(f"Error extracting {bucket_key}: {e}")
        return []


def process_timestep_to_training_sample(
    timestep: Dict[str, Any], metadata: Dict[str, Any], source_file: str
) -> Optional[Dict[str, Any]]:
    """Convert a single timestep to comprehensive training sample format."""
    try:
        # Extract features with validation
        battery_level = float(timestep.get("battery_before", 0.0))
        clean_energy_pct = float(timestep.get("clean_energy_pct", 0.0))

        # Extract system configuration from metadata
        battery_capacity_wh = float(metadata.get("battery_capacity_wh", 0.0))
        charge_rate_hours = float(metadata.get("charge_rate_hours", 0.0))
        task_interval_seconds = int(metadata.get("task_interval_seconds", 300))

        # Extract user requirements from metadata
        user_accuracy_req = float(metadata.get("user_accuracy_requirement", 0.905))
        user_latency_req = float(metadata.get("user_latency_requirement", 0.01))

        # Extract target labels
        optimal_model = str(timestep.get("model", ""))
        should_charge = bool(timestep.get("charged", False))
        outcome = str(timestep.get("outcome", ""))

        # Calculate derived features for better ML performance
        battery_percentage = (
            (battery_level / battery_capacity_wh) * 100
            if battery_capacity_wh > 0
            else 0.0
        )
        energy_efficiency_score = clean_energy_pct * (1 - battery_percentage / 100)

        return {
            # Core input features
            "battery_level": battery_level,
            "battery_percentage": battery_percentage,
            "clean_energy_percentage": clean_energy_pct,
            "battery_capacity_wh": battery_capacity_wh,
            "charge_rate_hours": charge_rate_hours,
            "task_interval_seconds": task_interval_seconds,
            "user_accuracy_requirement": user_accuracy_req,
            "user_latency_requirement": user_latency_req,
            # Derived features for ML
            "energy_efficiency_score": energy_efficiency_score,
            "time_to_full_charge": (battery_capacity_wh - battery_level)
            / (battery_capacity_wh / charge_rate_hours)
            if charge_rate_hours > 0
            else 0.0,
            # Target labels
            "optimal_model": optimal_model,
            "should_charge": should_charge,
            "outcome": outcome,
            # Metadata for traceability
            "source_file": source_file,
            "processing_timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        print(f"Error processing timestep: {e}")
        return None


def process_single_file(
    filepath: str,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    """Process a single metadata JSON file and extract bucket sequences."""
    print(f"Processing {filepath}...")

    data = load_metadata_file(filepath)
    if not data:
        return {
            "success": [],
            "success_small_miss": [],
            "most_clean_energy": [],
            "least_total_energy": [],
        }, {"error": "Failed to load file"}

    metadata = data.get("metadata", {})
    if not validate_metadata(metadata):
        print(f"Warning: Invalid metadata in {filepath}")
        return {
            "success": [],
            "success_small_miss": [],
            "most_clean_energy": [],
            "least_total_energy": [],
        }, {"error": "Invalid metadata"}

    # Extract sequences from each bucket
    bucket_mapping = {
        "success": "top_success",
        "success_small_miss": "top_success_small_miss",
        "most_clean_energy": "top_most_clean_energy",
        "least_total_energy": "top_least_total_energy",
    }

    buckets = {}
    for bucket_name, bucket_key in bucket_mapping.items():
        buckets[bucket_name] = extract_bucket_sequence(data, bucket_key)

    # Convert timesteps to training samples
    processed_buckets = {}
    file_stats = {"total_timesteps": 0, "valid_samples": 0, "invalid_samples": 0}

    for bucket_name, sequence in buckets.items():
        training_samples = []
        for timestep in sequence:
            file_stats["total_timesteps"] += 1
            sample = process_timestep_to_training_sample(
                timestep, metadata, Path(filepath).name
            )
            if sample:
                training_samples.append(sample)
                file_stats["valid_samples"] += 1
            else:
                file_stats["invalid_samples"] += 1
        processed_buckets[bucket_name] = training_samples

    return processed_buckets, file_stats


def balance_datasets(
    all_bucket_data: Dict[str, List[Dict[str, Any]]], random_seed: Optional[int] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """Balance datasets to have equal samples across all buckets."""
    if random_seed is not None:
        random.seed(random_seed)

    # Find minimum sample count across all buckets
    sample_counts = [len(samples) for samples in all_bucket_data.values() if samples]
    if not sample_counts:
        print("Warning: No samples found in any bucket!")
        return all_bucket_data

    min_samples = min(sample_counts)
    print(f"Balancing datasets to {min_samples} samples per bucket")

    # Randomly sample to balance each bucket
    balanced_data = {}
    for bucket_name, samples in all_bucket_data.items():
        if len(samples) >= min_samples:
            # Randomly sample to balance
            balanced_samples = random.sample(samples, min_samples)
        else:
            balanced_samples = samples.copy()
        balanced_data[bucket_name] = balanced_samples

    # Print sample counts
    for bucket_name, samples in balanced_data.items():
        print(f"  {bucket_name}: {len(samples)} samples")

    return balanced_data


def calculate_dataset_statistics(
    balanced_data: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Calculate comprehensive statistics for the balanced dataset."""
    stats = {
        "total_samples": sum(len(samples) for samples in balanced_data.values()),
        "bucket_stats": {},
        "global_feature_stats": {},
        "model_distribution": {},
        "outcome_distribution": {},
        "charge_decision_distribution": {"true": 0, "false": 0},
    }

    # Calculate per-bucket and global statistics
    all_samples = []
    for bucket_name, samples in balanced_data.items():
        bucket_stats = {
            "total_samples": len(samples),
            "model_counts": {},
            "outcome_counts": {},
            "charge_counts": {"true": 0, "false": 0},
            "feature_ranges": {},
        }

        for sample in samples:
            all_samples.append(sample)

            # Count models
            model = sample.get("optimal_model", "unknown")
            bucket_stats["model_counts"][model] = (
                bucket_stats["model_counts"].get(model, 0) + 1
            )
            stats["model_distribution"][model] = (
                stats["model_distribution"].get(model, 0) + 1
            )

            # Count outcomes
            outcome = sample.get("outcome", "unknown")
            bucket_stats["outcome_counts"][outcome] = (
                bucket_stats["outcome_counts"].get(outcome, 0) + 1
            )
            stats["outcome_distribution"][outcome] = (
                stats["outcome_distribution"].get(outcome, 0) + 1
            )

            # Count charges
            should_charge = sample.get("should_charge", False)
            charge_key = "true" if should_charge else "false"
            bucket_stats["charge_counts"][charge_key] += 1
            stats["charge_decision_distribution"][charge_key] += 1

        # Calculate feature ranges for this bucket
        if samples:
            numeric_features = [
                "battery_level",
                "battery_percentage",
                "clean_energy_percentage",
                "energy_efficiency_score",
                "time_to_full_charge",
            ]
            for feature in numeric_features:
                values = [sample.get(feature, 0) for sample in samples]
                bucket_stats["feature_ranges"][feature] = {
                    "min": min(values),
                    "max": max(values),
                    "mean": sum(values) / len(values),
                }

        stats["bucket_stats"][bucket_name] = bucket_stats

    # Calculate global feature statistics
    if all_samples:
        numeric_features = [
            "battery_level",
            "battery_percentage",
            "clean_energy_percentage",
            "energy_efficiency_score",
            "time_to_full_charge",
        ]
        for feature in numeric_features:
            values = [sample.get(feature, 0) for sample in all_samples]
            stats["global_feature_stats"][feature] = {
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "std": (
                    sum((x - sum(values) / len(values)) ** 2 for x in values)
                    / len(values)
                )
                ** 0.5,
            }

    return stats


def save_training_data(
    bucket_samples: Dict[str, List[Dict[str, Any]]],
    output_dir: str,
    stats: Dict[str, Any],
) -> None:
    """Save training data to JSON files with comprehensive metadata."""
    os.makedirs(output_dir, exist_ok=True)

    bucket_filenames = {
        "success": "success-training-data.json",
        "success_small_miss": "success_small_miss-training-data.json",
        "most_clean_energy": "most_clean_energy-training-data.json",
        "least_total_energy": "least_total_energy-training-data.json",
    }

    # Save individual bucket files
    for bucket_name, samples in bucket_samples.items():
        filename = bucket_filenames[bucket_name]
        filepath = os.path.join(output_dir, filename)

        # Comprehensive dataset metadata
        dataset_info = {
            "bucket_name": bucket_name,
            "total_samples": len(samples),
            "preprocessing_info": {
                "features": [
                    "battery_level",
                    "battery_percentage",
                    "clean_energy_percentage",
                    "battery_capacity_wh",
                    "charge_rate_hours",
                    "task_interval_seconds",
                    "user_accuracy_requirement",
                    "user_latency_requirement",
                    "energy_efficiency_score",
                    "time_to_full_charge",
                ],
                "labels": ["optimal_model", "should_charge", "outcome"],
                "derived_features": [
                    "battery_percentage",
                    "energy_efficiency_score",
                    "time_to_full_charge",
                ],
                "metadata_fields": ["source_file", "processing_timestamp"],
            },
            "bucket_statistics": stats["bucket_stats"].get(bucket_name, {}),
            "training_samples": samples,
        }

        with open(filepath, "w") as f:
            json.dump(dataset_info, f, indent=2)

        print(f"Saved {len(samples)} samples to {filepath}")

    # Save comprehensive statistics file
    stats_filepath = os.path.join(output_dir, "dataset-statistics.json")
    with open(stats_filepath, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved comprehensive statistics to {stats_filepath}")


def main():
    """Main preprocessing pipeline with enhanced features."""
    parser = argparse.ArgumentParser(
        description="Preprocess beam search results into training data"
    )
    parser.add_argument(
        "--random-seed", type=int, default=None, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--results-dir", type=str, default="results", help="Results directory path"
    )
    parser.add_argument(
        "--output-dir", type=str, default="training-data", help="Output directory path"
    )
    args = parser.parse_args()

    print("Starting comprehensive beam bucket data preprocessing...")
    print("=" * 60)
    print(f"Random seed: {args.random_seed}")
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")

    # Setup paths
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    if not results_dir.exists():
        print(f"Error: Results directory {results_dir} does not exist!")
        return

    # Find all metadata JSON files
    json_files = list(results_dir.glob("*-metadata.json"))
    print(f"Found {len(json_files)} JSON files to process")

    if not json_files:
        print("No metadata files found. Please run simulations first.")
        return

    # Process all files and accumulate bucket data
    all_bucket_data = {
        "success": [],
        "success_small_miss": [],
        "most_clean_energy": [],
        "least_total_energy": [],
    }

    processing_stats = {
        "files_processed": 0,
        "files_failed": 0,
        "total_timesteps": 0,
        "valid_samples": 0,
        "invalid_samples": 0,
    }

    for filepath in sorted(json_files):
        file_buckets, file_stats = process_single_file(str(filepath))

        if "error" in file_stats:
            processing_stats["files_failed"] += 1
            continue

        processing_stats["files_processed"] += 1
        processing_stats["total_timesteps"] += file_stats["total_timesteps"]
        processing_stats["valid_samples"] += file_stats["valid_samples"]
        processing_stats["invalid_samples"] += file_stats["invalid_samples"]

        # Accumulate data from each file
        for bucket_name, samples in file_buckets.items():
            all_bucket_data[bucket_name].extend(samples)

    print("\nProcessing Summary:")
    print(f"  Files processed: {processing_stats['files_processed']}")
    print(f"  Files failed: {processing_stats['files_failed']}")
    print(f"  Total timesteps: {processing_stats['total_timesteps']}")
    print(f"  Valid samples: {processing_stats['valid_samples']}")
    print(f"  Invalid samples: {processing_stats['invalid_samples']}")

    print("\nTotal samples before balancing:")
    for bucket_name, samples in all_bucket_data.items():
        print(f"  {bucket_name}: {len(samples)} samples")

    # Balance datasets
    print("\nBalancing datasets...")
    balanced_data = balance_datasets(all_bucket_data, args.random_seed)

    # Calculate comprehensive statistics
    print("\nCalculating dataset statistics...")
    stats = calculate_dataset_statistics(balanced_data)

    # Save training data
    print(f"\nSaving training datasets to {output_dir}/...")
    save_training_data(balanced_data, str(output_dir), stats)

    print("\n" + "=" * 60)
    print("Data preprocessing completed successfully!")
    print(f"Output directory: {output_dir.absolute()}")
    print("Generated training files:")
    print("  - training-data/success-training-data.json")
    print("  - training-data/success_small_miss-training-data.json")
    print("  - training-data/most_clean_energy-training-data.json")
    print("  - training-data/least_total_energy-training-data.json")
    print("  - training-data/dataset-statistics.json")

    # Print summary statistics
    print("\nDataset Summary:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Unique models: {len(stats['model_distribution'])}")
    print(f"  Unique outcomes: {len(stats['outcome_distribution'])}")
    print(f"  Charge decisions: {stats['charge_decision_distribution']}")

    print("\nGlobal Feature Statistics:")
    for feature, feature_stats in stats["global_feature_stats"].items():
        print(
            f"  {feature}: min={feature_stats['min']:.3f}, max={feature_stats['max']:.3f}, mean={feature_stats['mean']:.3f}"
        )


if __name__ == "__main__":
    main()
