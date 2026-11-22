"""Explicit Forward-Time Central-Space (FTCS) solver for the 2D heat equation.

Mathematical Formulation & Derivation:
-------------------------------------
The continuous 2D heat diffusion equation without internal sources is:
    du/dt = alpha * (d^2u/dx^2 + d^2u/dy^2)

Discretizing onto a Cartesian grid with spatial steps dx, dy and time step dt:
1. Temporal derivative (Forward Euler, first-order accurate O(dt)):
   du/dt |_{i,j}^n = (u_{i,j}^{n+1} - u_{i,j}^n) / dt + O(dt)

2. Spatial second derivatives (Central differences, second-order O(dx^2 + dy^2)):
   d^2u/dx^2 |_{i,j}^n = (u_{i+1,j}^n - 2*u_{i,j}^n + u_{i-1,j}^n) / (dx^2) + O(dx^2)
   d^2u/dy^2 |_{i,j}^n = (u_{i,j+1}^n - 2*u_{i,j}^n + u_{i,j-1}^n) / (dy^2) + O(dy^2)

Substituting these discrete approximations into the PDE yields the update equation:
    u_{i,j}^{n+1} = u_{i,j}^n + r_x * (u_{i+1,j}^n - 2*u_{i,j}^n + u_{i-1,j}^n)
                             + r_y * (u_{i,j+1}^n - 2*u_{i,j}^n + u_{i,j-1}^n)

where the dimensionless mesh Fourier numbers are:
    r_x = (alpha * dt) / (dx^2),   r_y = (alpha * dt) / (dy^2)

Physical Meaning of Terms:
-------------------------
- u_{i,j}^{n+1}: Future temperature at spatial coordinate (x_i, y_j) at time t^{n+1}.
- u_{i,j}^n: Current temperature at spatial coordinate (x_i, y_j) at time t^n.
- (u_{i+1,j}^n - 2*u_{i,j}^n + u_{i-1,j}^n): Discrete spatial curvature along the
  x-axis. By Fourier's law (q = -k * grad(u)), positive curvature means the node is
  colder than the average of horizontal neighbors, drawing net conductive heat flux.
- (u_{i,j+1}^n - 2*u_{i,j}^n + u_{i,j-1}^n): Discrete spatial curvature along the
  y-axis, representing net conductive heat influx from vertical neighbors.
- r_x, r_y: Dimensionless rate ratios quantifying thermal diffusion rate relative
  to the numerical storage capacity of the grid cell per time step.

Stability Criterion (Von Neumann Analysis):
-------------------------------------------
Expressing error as Fourier modes e^{I*(kx*x + ky*y)}, amplification factor xi is:
    xi = 1 - 4*r_x*sin^2(kx*dx / 2) - 4*r_y*sin^2(ky*dy / 2)

Numerical stability requires |xi| <= 1 for all wavenumbers (kx, ky).
The critical lower bound xi >= -1 at Nyquist frequencies (sin^2 = 1) yields:
    1 - 4*r_x - 4*r_y >= -1  ==>  r_x + r_y <= 0.5

Physical Interpretation of Stability:
Rewriting the update equation as a convex combination:
    u_{i,j}^{n+1} = (1 - 2*r_x - 2*r_y)*u_{i,j}^n + r_x*(u_{i+1,j}^n + u_{i-1,j}^n)
                                                   + r_y*(u_{i,j+1}^n + u_{i,j-1}^n)
When (r_x + r_y) <= 0.5, all coefficients are non-negative and sum to 1.0, satisfying
the discrete maximum principle (temperatures remain bounded within initial extrema).
When (r_x + r_y) > 0.5, the self-weight becomes negative, meaning a hotter node becomes
colder than its neighbors in the next step, causing unbounded oscillatory blowup.
"""

from dataclasses import dataclass

import numpy as np

from physics_ml.config import SimulationConfig
from physics_ml.physics.boundary import BoundaryCondition, ZeroDirichletBoundary
from physics_ml.simulation.grid import Grid2D


class NumericalInstabilityError(ValueError):
    """Raised when discretization violates the 2D FTCS stability criterion."""

    def __init__(self, rx: float, ry: float, dt: float, max_dt: float) -> None:
        total_r = rx + ry
        message = (
            f"2D FTCS numerical instability detected: rx + ry = {total_r:.4f} > 0.5 "
            f"(rx={rx:.4f}, ry={ry:.4f}). Given time step dt={dt:.6e}s exceeds maximum "
            f"stable time step dt_max={max_dt:.6e}s. "
            "Reduce time_step or coarsen the spatial grid resolution."
        )
        super().__init__(message)
        self.rx = rx
        self.ry = ry
        self.total_r = total_r
        self.dt = dt
        self.max_dt = max_dt


@dataclass(frozen=True)
class SimulationResult:
    """Encapsulates the output of a 2D heat diffusion numerical simulation.

    Attributes:
        times: 1D array of physical time values corresponding to recorded states [s].
        states: 3D array of temperature fields of shape (num_recorded, Nx, Ny).
        config: SimulationConfig used to generate the result.
    """

    times: np.ndarray
    states: np.ndarray
    config: SimulationConfig

    @property
    def final_state(self) -> np.ndarray:
        """The temperature field at the final recorded time step."""
        return self.states[-1]

    @property
    def final_time(self) -> float:
        """The final recorded simulation time."""
        return float(self.times[-1])


class FTCSSolver2D:
    """Explicit 2D Forward-Time Central-Space finite difference numerical solver."""

    def __init__(
        self,
        config: SimulationConfig,
        boundary_condition: BoundaryCondition | None = None,
    ) -> None:
        """Initialize the 2D FTCS solver with configuration and boundary conditions.

        Args:
            config: Simulation configuration specifying domain, diffusivity, and steps.
            boundary_condition: Boundary condition handler. Defaults to ZeroDirichlet.

        Raises:
            NumericalInstabilityError: If rx + ry > 0.5.
        """
        self.config = config
        self.grid = Grid2D.from_config(config)
        self.boundary_condition = boundary_condition or ZeroDirichletBoundary()

        self.rx = config.fourier_number_x
        self.ry = config.fourier_number_y
        self.total_r = self.rx + self.ry

        # Strictly enforce stability condition rx + ry <= 0.5
        if self.total_r > 0.5:
            raise NumericalInstabilityError(
                rx=self.rx,
                ry=self.ry,
                dt=config.time_step,
                max_dt=config.max_stable_time_step,
            )

    def step(self, u: np.ndarray) -> np.ndarray:
        """Advance the 2D temperature field by a single time increment dt.

        Utilizes vectorized NumPy slicing over the interior domain [1:-1, 1:-1]
        for maximal computational efficiency.

        Args:
            u: Current 2D temperature field of shape (Nx, Ny).

        Returns:
            Updated 2D temperature field of shape (Nx, Ny) at t + dt.
        """
        u_next = u.copy()

        # Vectorized 2D Laplacian operator applied to interior nodes
        laplacian_x = u[2:, 1:-1] - 2.0 * u[1:-1, 1:-1] + u[:-2, 1:-1]
        laplacian_y = u[1:-1, 2:] - 2.0 * u[1:-1, 1:-1] + u[1:-1, :-2]

        u_next[1:-1, 1:-1] = (
            u[1:-1, 1:-1] + (self.rx * laplacian_x) + (self.ry * laplacian_y)
        )

        # Enforce boundary conditions on exterior edges
        return self.boundary_condition.apply(u_next)

    def solve(
        self,
        u0: np.ndarray,
        num_steps: int | None = None,
        record_interval: int = 1,
    ) -> SimulationResult:
        """Integrate the 2D heat equation forward in time from an initial temperature.

        Args:
            u0: Initial temperature field at t = 0 of shape (Nx, Ny).
            num_steps: Number of integration steps. Defaults to config.num_time_steps.
            record_interval: Frequency of saving states (e.g. 1 records all steps).

        Returns:
            SimulationResult containing recorded physical times and state arrays.
        """
        if u0.shape != self.grid.shape:
            raise ValueError(
                f"Initial condition shape {u0.shape} != grid shape {self.grid.shape}"
            )
        if record_interval < 1:
            raise ValueError(f"record_interval must be >= 1, got {record_interval}")

        total_steps = self.config.num_time_steps if num_steps is None else num_steps
        if total_steps < 0:
            raise ValueError(f"num_steps must be non-negative, got {total_steps}")

        # Ensure initial field complies with boundary conditions
        current_u = self.boundary_condition.apply(u0.copy())

        recorded_times = [0.0]
        recorded_states = [current_u.copy()]

        for step_idx in range(1, total_steps + 1):
            current_u = self.step(current_u)
            if step_idx % record_interval == 0 or step_idx == total_steps:
                recorded_times.append(step_idx * self.config.time_step)
                recorded_states.append(current_u.copy())

        return SimulationResult(
            times=np.array(recorded_times, dtype=np.float64),
            states=np.array(recorded_states, dtype=np.float64),
            config=self.config,
        )
