#!/usr/bin/env python3
"""
Quick test of power benchmarking with single model.
"""

import logging
from pathlib import Path

from src.power_profiler import PowerProfiler
from src.logging_config import setup_logging


def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    profiler = PowerProfiler()

    # Test with single image and model
    image1 = Path("benchmark-images/image1.png")
    if not image1.exists():
        logger.error(f"Benchmark image not found: {image1}")
        return 1

    # Test single model
    profile = profiler.benchmark_model_power("YOLOv10", "N", str(image1), iterations=2)

    print("Quick test results:")
    print(f"Model Power: {profile['model_power_mw']:.2f} mW")
    print(f"Baseline: {profile['baseline_power_mw']:.2f} mW")
    print(f"Avg Inference: {profile['avg_inference_power_mw']:.2f} mW")

    return 0


if __name__ == "__main__":
    exit(main())
