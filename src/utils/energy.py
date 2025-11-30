"""Energy and battery unit conversion utilities."""

from typing import Union


def energy_joules_to_wh(joules: float) -> float:
    """Convert energy from Joules to Watt-hours.
    
    Args:
        joules: Energy in Joules
        
    Returns:
        Energy in Watt-hours
    """
    return joules / 3600.0


def energy_wh_to_percent(energy_wh: float, capacity_wh: float) -> float:
    """Convert energy from Watt-hours to battery percentage (0-100).
    
    Args:
        energy_wh: Energy in Watt-hours
        capacity_wh: Total battery capacity in Watt-hours
        
    Returns:
        Battery percentage (0-100)
    """
    if capacity_wh <= 0:
        return 0.0
    return (energy_wh / capacity_wh) * 100.0


def energy_percent_to_wh(percent: float, capacity_wh: float) -> float:
    """Convert battery percentage (0-100) to energy in Watt-hours.
    
    Args:
        percent: Battery percentage (0-100)
        capacity_wh: Total battery capacity in Watt-hours
        
    Returns:
        Energy in Watt-hours
    """
    return (percent / 100.0) * capacity_wh


def calculate_inference_energy_wh(power_watts: float, latency_s: float) -> float:
    """Calculate energy consumed per inference in Watt-hours.
    
    Args:
        power_watts: Power draw in Watts during inference
        latency_s: Inference latency in seconds
        
    Returns:
        Energy consumed in Watt-hours
    """
    # Energy = Power × Time (in Joules)
    energy_joules = power_watts * latency_s
    # Convert to Watt-hours
    return energy_joules_to_wh(energy_joules)

