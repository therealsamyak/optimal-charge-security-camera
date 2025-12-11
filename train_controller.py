#!/usr/bin/env python3
"""
Train Unified Custom Controller
Entry point for training unified controller.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.controller_trainer import main

if __name__ == "__main__":
    sys.exit(main())
