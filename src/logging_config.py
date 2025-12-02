import logging
import sys
from pathlib import Path


def setup_logging():
    """Setup error-focused logging configuration."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "error.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    return logging.getLogger(__name__)
