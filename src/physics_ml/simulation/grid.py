"""Spatial grid representation and discretization for 2D computational domains."""

from dataclasses import dataclass

import numpy as np

from physics_ml.config import SimulationConfig


@dataclass(frozen=True)
class Grid2D:
    """Discretized 2D spatial grid for numerical simulation.

    Attributes:
        x: 1D array of node coordinates along x-axis of length Nx.
        y: 1D array of node coordinates along y-axis of length Ny.
        X: 2D meshgrid array of x-coordinates with shape (Nx, Ny).
        Y: 2D meshgrid array of y-coordinates with shape (Nx, Ny).
        dx: Spatial grid spacing along x-axis.
        dy: Spatial grid spacing along y-axis.
        shape: Tuple of spatial grid dimensions (Nx, Ny).
    """

    x: np.ndarray
    y: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    dx: float
    dy: float
    shape: tuple[int, int]

    @classmethod
    def from_config(cls, config: SimulationConfig) -> "Grid2D":
        """Instantiate a 2D grid from configuration parameters.

        Args:
            config: Simulation configuration with dimensions and resolution.

        Returns:
            Configured Grid2D instance with coordinate arrays.
        """
        x = np.linspace(0.0, config.plate_width, config.grid_resolution_x)
        y = np.linspace(0.0, config.plate_height, config.grid_resolution_y)
        x_mesh, y_mesh = np.meshgrid(x, y, indexing="ij")
        dx = float(x[1] - x[0])
        dy = float(y[1] - y[0])
        return cls(
            x=x,
            y=y,
            X=x_mesh,
            Y=y_mesh,
            dx=dx,
            dy=dy,
            shape=(config.grid_resolution_x, config.grid_resolution_y),
        )

    def zeros(self) -> np.ndarray:
        """Create a zero-filled 2D field of shape (Nx, Ny)."""
        return np.zeros(self.shape, dtype=np.float64)
