"""
Sensors package for battery and energy monitoring.

This package provides interfaces for monitoring battery status and energy cleanliness.
Colleague should implement the actual hardware integration in the respective modules.
"""

from .battery import BatterySensor
from .energy import EnergySensor

__all__ = ["BatterySensor", "EnergySensor"]
