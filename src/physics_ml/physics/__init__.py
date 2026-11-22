"""Physics module: PDE formulations, physical constants, and boundary conditions."""

from physics_ml.physics.analytical import (
    analytical_decay_rate,
    single_mode_analytical_solution,
)
from physics_ml.physics.boundary import BoundaryCondition, ZeroDirichletBoundary

__all__ = [
    "BoundaryCondition",
    "ZeroDirichletBoundary",
    "analytical_decay_rate",
    "single_mode_analytical_solution",
]
