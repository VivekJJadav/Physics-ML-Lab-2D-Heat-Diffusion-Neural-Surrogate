"""Boundary condition protocols and implementations for 2D spatial domains.

Defines the extensible boundary condition interface and the baseline
zero Dirichlet condition on all four edges of the 2D rectangular plate.
"""

from abc import ABC, abstractmethod

import numpy as np


class BoundaryCondition(ABC):
    """Abstract base class defining boundary condition enforcement on a 2D field."""

    @abstractmethod
    def apply(self, u: np.ndarray) -> np.ndarray:
        """Apply boundary conditions in-place or return updated field.

        Args:
            u: 2D temperature array of shape (Nx, Ny).

        Returns:
            2D temperature array with boundary conditions enforced.
        """
        ...


class ZeroDirichletBoundary(BoundaryCondition):
    """Zero Dirichlet boundary conditions on all four plate edges.

    Enforces u(0, y) = u(Lx, y) = u(x, 0) = u(x, Ly) = 0.
    """

    def apply(self, u: np.ndarray) -> np.ndarray:
        """Enforce u = 0 on all four outer boundaries of the 2D grid.

        Args:
            u: 2D temperature array of shape (Nx, Ny).

        Returns:
            Array `u` with outer boundary values clamped to 0.0.
        """
        u[0, :] = 0.0  # x = 0 (bottom edge in coordinate space)
        u[-1, :] = 0.0  # x = Lx (top edge)
        u[:, 0] = 0.0  # y = 0 (left edge)
        u[:, -1] = 0.0  # y = Ly (right edge)
        return u
