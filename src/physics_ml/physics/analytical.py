"""Analytical solutions for the 2D heat diffusion equation.

Provides exact closed-form benchmark solutions for verifying numerical PDE solvers
and evaluating surrogate model accuracy.
"""

import numpy as np

from physics_ml.config import SimulationConfig


def analytical_decay_rate(config: SimulationConfig) -> float:
    """Compute the exponential decay rate lambda for the (1, 1) Fourier mode.

    The continuous decay rate is:
        lambda = alpha * pi^2 * (1 / Lx^2 + 1 / Ly^2)

    Args:
        config: Simulation configuration specifying dimensions and diffusivity.

    Returns:
        Temporal decay constant [1/s].
    """
    inv_lx2 = 1.0 / (config.plate_width**2)
    inv_ly2 = 1.0 / (config.plate_height**2)
    return float(config.thermal_diffusivity * (np.pi**2) * (inv_lx2 + inv_ly2))


def single_mode_analytical_solution(
    x: np.ndarray,
    y: np.ndarray,
    t: float,
    config: SimulationConfig,
) -> np.ndarray:
    """Compute the single-mode analytical solution for 2D heat diffusion.

    The exact solution on [0, Lx] x [0, Ly] with zero Dirichlet boundary conditions is:
        u(x, y, t) = sin(pi * x / Lx) * sin(pi * y / Ly) * exp(-lambda * t)
    where:
        lambda = alpha * pi^2 * (1 / Lx^2 + 1 / Ly^2)

    Args:
        x: Coordinate array along x. Can be 1D (Nx,) or 2D (Nx, Ny).
        y: Coordinate array along y. Can be 1D (Ny,) or 2D (Nx, Ny).
        t: Non-negative physical time [s] at which to evaluate the solution.
        config: Simulation configuration containing dimensions and diffusivity.

    Returns:
        2D array of exact temperatures with shape (Nx, Ny).
    """
    if t < 0.0:
        raise ValueError(f"Evaluation time t must be non-negative, got {t}")

    # If 1D coordinate arrays were passed, broadcast into 2D grid
    if x.ndim == 1 and y.ndim == 1:
        x_grid, y_grid = np.meshgrid(x, y, indexing="ij")
    elif x.ndim == 2 and y.ndim == 2:
        x_grid, y_grid = x, y
    else:
        raise ValueError(
            f"x and y must both be 1D or both 2D, got x.ndim={x.ndim}, y.ndim={y.ndim}"
        )

    decay = np.exp(-analytical_decay_rate(config) * t)
    spatial_mode = np.sin(np.pi * x_grid / config.plate_width) * np.sin(
        np.pi * y_grid / config.plate_height
    )
    return spatial_mode * decay
