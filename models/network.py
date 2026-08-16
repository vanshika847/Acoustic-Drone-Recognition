"""Backward-compatible import for the acoustic drone model."""

from models.acoustic_drone_model import AcousticDroneModel


# Backward-compatible name for existing training/inference code.
AcousticDroneNet = AcousticDroneModel


__all__ = [
    "AcousticDroneModel",
    "AcousticDroneNet",
]