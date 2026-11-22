"""Central configuration module for scientific ML experiments.

This module provides structured configuration schemas for setting up 2D heat diffusion
numerical simulations and machine learning experiments.

Note on Default Values:
    The default values defined below represent initial, non-dimensionalized or canonical
    experimental settings for baseline verification. They do NOT represent universally
    valid constants for all materials or operational regimes. For instance, physical
    metals exhibit specific thermal diffusivities (e.g., pure copper ~ 1.11e-4 m²/s,
    aluminum ~ 9.7e-5 m²/s at room temperature), whereas non-dimensionalized equations
    frequently scale parameters to unity. In subsequent milestones, problem-specific
    configurations will override these defaults.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration parameters for 2D heat diffusion simulation.

    Defines the spatial geometry, grid resolution, material diffusion properties,
    and temporal discretization for numerical integration and surrogate modeling.

    Attributes:
        plate_width: Physical dimension of the rectangular plate along the x-axis [m].
            Default: 1.0.
        plate_height: Physical dimension of the rectangular plate along the y-axis [m].
            Default: 1.0.
        grid_resolution_x: Number of spatial discretization points along the x-axis.
            Must be at least 3 to support interior and boundary evaluations.
            Default: 51.
        grid_resolution_y: Number of spatial discretization points along the y-axis.
            Must be at least 3 to support interior and boundary evaluations.
            Default: 51.
        thermal_diffusivity: Material thermal diffusivity (alpha) [m²/s].
            Controls the rate of temperature equalization. Must be strictly positive.
            Default: 0.01.
        simulation_duration: Total physical time to advance the simulation [s].
            Must be strictly positive.
            Default: 1.0.
        time_step: Discrete temporal advancement step size (delta_t) [s].
            Must be strictly positive and no greater than simulation_duration.
            Default: 0.005.
    """

    plate_width: float = 1.0
    plate_height: float = 1.0
    grid_resolution_x: int = 51
    grid_resolution_y: int = 51
    thermal_diffusivity: float = 0.01
    simulation_duration: float = 1.0
    time_step: float = 0.005

    def __post_init__(self) -> None:
        """Validate configuration parameters for physical and numerical sanity."""
        if self.plate_width <= 0.0:
            raise ValueError(f"plate_width must be positive, got {self.plate_width}")
        if self.plate_height <= 0.0:
            raise ValueError(f"plate_height must be positive, got {self.plate_height}")
        if self.grid_resolution_x < 3:
            raise ValueError(
                f"grid_resolution_x must be >= 3, got {self.grid_resolution_x}"
            )
        if self.grid_resolution_y < 3:
            raise ValueError(
                f"grid_resolution_y must be >= 3, got {self.grid_resolution_y}"
            )
        if self.thermal_diffusivity <= 0.0:
            raise ValueError(
                f"thermal_diffusivity must be positive, got {self.thermal_diffusivity}"
            )
        if self.simulation_duration <= 0.0:
            raise ValueError(
                f"simulation_duration must be positive, got {self.simulation_duration}"
            )
        if self.time_step <= 0.0:
            raise ValueError(f"time_step must be positive, got {self.time_step}")
        if self.time_step > self.simulation_duration:
            raise ValueError(
                f"time_step ({self.time_step}) cannot exceed simulation_duration "
                f"({self.simulation_duration})"
            )

    @property
    def dx(self) -> float:
        """Spatial discretization step along the x-axis: dx = plate_width / (Nx - 1)."""
        return self.plate_width / (self.grid_resolution_x - 1)

    @property
    def dy(self) -> float:
        """Spatial step along the y-axis: dy = plate_height / (Ny - 1)."""
        return self.plate_height / (self.grid_resolution_y - 1)

    @property
    def num_time_steps(self) -> int:
        """Total number of discrete time integration steps."""
        return int(self.simulation_duration / self.time_step)

    @property
    def fourier_number_x(self) -> float:
        """Mesh Fourier number along x-axis: alpha * dt / (dx^2).

        In standard explicit finite-difference (FTCS) schemes, stability requires
        (fourier_number_x + fourier_number_y) <= 0.5.
        """
        return (self.thermal_diffusivity * self.time_step) / (self.dx**2)

    @property
    def fourier_number_y(self) -> float:
        """Mesh Fourier number along y-axis: alpha * dt / (dy^2)."""
        return (self.thermal_diffusivity * self.time_step) / (self.dy**2)

    @property
    def total_fourier_number(self) -> float:
        """Total mesh Fourier number rx + ry."""
        return self.fourier_number_x + self.fourier_number_y

    @property
    def is_ftcs_stable(self) -> bool:
        """Check whether current spatial-temporal parameters satisfy 2D FTCS stability.

        The Von Neumann stability criterion requires: rx + ry <= 0.5.
        """
        return self.total_fourier_number <= 0.5

    @property
    def max_stable_time_step(self) -> float:
        """Maximum allowable time step (dt) to ensure 2D FTCS numerical stability."""
        spatial_factor = (1.0 / self.dx**2) + (1.0 / self.dy**2)
        return 0.5 / (self.thermal_diffusivity * spatial_factor)


# Initial default configuration instance for quick access and testing
DEFAULT_SIMULATION_CONFIG = SimulationConfig()
