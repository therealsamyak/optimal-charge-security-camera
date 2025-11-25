"""Cache management utilities for expensive computations."""

import json
import pickle
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional
from loguru import logger


CACHE_DIR = Path(".cache")
ENERGY_CACHE_DIR = CACHE_DIR / "energy"
ORACLE_CACHE_DIR = CACHE_DIR / "oracle"


def ensure_cache_dirs():
    """Ensure cache directories exist."""
    CACHE_DIR.mkdir(exist_ok=True)
    ENERGY_CACHE_DIR.mkdir(exist_ok=True)
    ORACLE_CACHE_DIR.mkdir(exist_ok=True)
    logger.debug(f"Cache directories ensured: {CACHE_DIR}")


def get_cache_key(*args, **kwargs) -> str:
    """Generate a cache key from arguments."""
    # Create a deterministic string representation
    key_data = {"args": args, "kwargs": sorted(kwargs.items()) if kwargs else {}}
    key_str = json.dumps(key_data, sort_keys=True, default=str)
    # Hash it to get a fixed-length key
    return hashlib.md5(key_str.encode()).hexdigest()


def get_oracle_cache_path(
    date: str,
    controller_type: str,
    accuracy_threshold: float,
    latency_threshold: float,
    initial_battery: float,
    model_data_hash: str,
) -> Path:
    """Get cache file path for Oracle optimization results."""
    ensure_cache_dirs()
    cache_key = get_cache_key(
        date,
        controller_type,
        accuracy_threshold,
        latency_threshold,
        initial_battery,
        model_data_hash,
    )
    return ORACLE_CACHE_DIR / f"{cache_key}.pkl"


def load_oracle_cache(cache_path: Path) -> Optional[Dict[Any, Any]]:
    """Load Oracle optimization results from cache."""
    if not cache_path.exists():
        return None

    try:
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        logger.info(f"Oracle cache HIT: {cache_path.name}")
        return data
    except Exception as e:
        logger.warning(f"Failed to load Oracle cache {cache_path}: {e}")
        return None


def save_oracle_cache(
    cache_path: Path, decisions: Dict[Any, Any], metadata: Dict[str, Any]
) -> None:
    """Save Oracle optimization results to cache."""
    ensure_cache_dirs()
    try:
        cache_data = {"decisions": decisions, "metadata": metadata}
        with open(cache_path, "wb") as f:
            pickle.dump(cache_data, f)
        logger.info(f"Oracle cache SAVE: {cache_path.name}")
    except Exception as e:
        logger.warning(f"Failed to save Oracle cache {cache_path}: {e}")


def get_energy_cache_path(date: str) -> Path:
    """Get cache file path for energy data."""
    ensure_cache_dirs()
    return ENERGY_CACHE_DIR / f"{date}.json"


def load_energy_cache(cache_path: Path) -> Optional[Dict[str, float]]:
    """Load energy data from cache."""
    if not cache_path.exists():
        return None

    try:
        with open(cache_path, "r") as f:
            data = json.load(f)
        # Convert string keys back to datetime objects (stored as ISO strings)
        result = {}
        from datetime import datetime

        for k, v in data.items():
            result[datetime.fromisoformat(k)] = v
        logger.info(f"Energy cache HIT: {cache_path.name}")
        return result
    except Exception as e:
        logger.warning(f"Failed to load energy cache {cache_path}: {e}")
        return None


def save_energy_cache(cache_path: Path, data: Dict[Any, float]) -> None:
    """Save energy data to cache."""
    ensure_cache_dirs()
    try:
        # Convert datetime keys to ISO strings for JSON serialization
        json_data = {}
        for k, v in data.items():
            if hasattr(k, "isoformat"):
                json_data[k.isoformat()] = v
            else:
                json_data[str(k)] = v

        with open(cache_path, "w") as f:
            json.dump(json_data, f)
        logger.info(f"Energy cache SAVE: {cache_path.name}")
    except Exception as e:
        logger.warning(f"Failed to save energy cache {cache_path}: {e}")


def clear_cache(cache_type: Optional[str] = None) -> None:
    """Clear cache files.

    Args:
        cache_type: If None, clear all caches. Otherwise "energy" or "oracle"
    """
    if cache_type is None:
        # Clear all
        import shutil

        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
            logger.info("Cleared all caches")
    elif cache_type == "energy":
        if ENERGY_CACHE_DIR.exists():
            import shutil

            shutil.rmtree(ENERGY_CACHE_DIR)
            logger.info("Cleared energy cache")
    elif cache_type == "oracle":
        if ORACLE_CACHE_DIR.exists():
            import shutil

            shutil.rmtree(ORACLE_CACHE_DIR)
            logger.info("Cleared Oracle cache")
    else:
        logger.warning(f"Unknown cache type: {cache_type}")


def get_cache_stats() -> Dict[str, Any]:
    """Get statistics about cache usage."""
    stats = {
        "energy_cache_files": 0,
        "oracle_cache_files": 0,
        "total_cache_size_mb": 0.0,
    }

    if ENERGY_CACHE_DIR.exists():
        energy_files = list(ENERGY_CACHE_DIR.glob("*.json"))
        stats["energy_cache_files"] = len(energy_files)
        stats["total_cache_size_mb"] += sum(f.stat().st_size for f in energy_files) / (
            1024 * 1024
        )

    if ORACLE_CACHE_DIR.exists():
        oracle_files = list(ORACLE_CACHE_DIR.glob("*.pkl"))
        stats["oracle_cache_files"] = len(oracle_files)
        stats["total_cache_size_mb"] += sum(f.stat().st_size for f in oracle_files) / (
            1024 * 1024
        )

    return stats
