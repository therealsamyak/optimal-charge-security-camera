"""
Controller package for intelligent model selection and battery management.

This package provides the intelligent controller that balances battery level,
energy cleanliness, and user requirements to select optimal YOLOv10 models.
"""

from .intelligent_controller import (
    ModelController,
    ControllerModelProfile,
    ControllerDecision,
)

__all__ = ["ModelController", "ControllerModelProfile", "ControllerDecision"]
