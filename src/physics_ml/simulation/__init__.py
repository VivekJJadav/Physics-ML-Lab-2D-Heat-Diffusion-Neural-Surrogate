"""Simulation module: Numerical solvers and spatial-temporal discretization routines."""

from physics_ml.simulation.ftcs import (
    FTCSSolver2D,
    NumericalInstabilityError,
    SimulationResult,
)
from physics_ml.simulation.grid import Grid2D

__all__ = [
    "FTCSSolver2D",
    "Grid2D",
    "NumericalInstabilityError",
    "SimulationResult",
]
