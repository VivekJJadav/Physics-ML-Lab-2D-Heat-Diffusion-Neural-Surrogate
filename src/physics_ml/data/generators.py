"""Initial condition generators for 2D heat diffusion simulations.

Provides two parameterized generators creating initial temperature fields on
[0, Lx] x [0, Ly] satisfying zero Dirichlet boundary conditions:
1. FourierModeGenerator: Superposition of orthogonal 2D Fourier sine modes.
2. GaussianSourceGenerator: Localized 2D Gaussian thermal sources.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from physics_ml.physics.boundary import ZeroDirichletBoundary
from physics_ml.simulation.grid import Grid2D


class InitialConditionGenerator(ABC):
    """Abstract base class for 2D initial temperature field generators."""

    @abstractmethod
    def generate(
        self,
        grid: Grid2D,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Generate initial temperature field u0(x, y) and its parameters.

        Args:
            grid: 2D spatial grid coordinate representation.
            rng: NumPy random number generator for reproducible sampling.

        Returns:
            Tuple of:
                - 2D temperature array u0 of shape (Nx, Ny).
                - Dictionary containing generative parameters and metadata.
        """
        ...


class FourierModeGenerator(InitialConditionGenerator):
    """Generates initial conditions as random superpositions of Fourier sine modes.

    The continuous formulation on [0, Lx] x [0, Ly] is:
        u0(x, y) = sum_{k=1}^K A_k * sin(m_k * pi * x / Lx) * sin(n_k * pi * y / Ly)

    Since sin(0) = sin(k * pi) = 0, every mode identically satisfies zero Dirichlet
    boundary conditions on all four edges without artificial boundary clamping.

    Attributes:
        min_mode: Minimum spatial frequency index (default: 1).
        max_mode: Maximum spatial frequency index (default: 4).
        min_active_modes: Minimum number of active modes to superimpose (default: 1).
        max_active_modes: Maximum number of active modes to superimpose (default: 4).
        target_peak_range: Min and max scaling for the peak temperature value.
    """

    def __init__(
        self,
        min_mode: int = 1,
        max_mode: int = 4,
        min_active_modes: int = 1,
        max_active_modes: int = 4,
        target_peak_range: tuple[float, float] = (0.5, 1.0),
    ) -> None:
        if min_mode < 1 or max_mode < min_mode:
            raise ValueError(
                f"Invalid mode indices: min_mode={min_mode}, max_mode={max_mode}"
            )
        if min_active_modes < 1 or max_active_modes < min_active_modes:
            raise ValueError(
                f"Invalid active mode count: min={min_active_modes}, "
                f"max={max_active_modes}"
            )
        if target_peak_range[0] <= 0 or target_peak_range[1] < target_peak_range[0]:
            raise ValueError(f"Invalid target_peak_range: {target_peak_range}")

        self.min_mode = min_mode
        self.max_mode = max_mode
        self.min_active_modes = min_active_modes
        self.max_active_modes = max_active_modes
        self.target_peak_range = target_peak_range
        self.bc = ZeroDirichletBoundary()

    def generate(
        self,
        grid: Grid2D,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Generate a randomized superposition of Fourier sine modes."""
        all_modes = [
            (m, n)
            for m in range(self.min_mode, self.max_mode + 1)
            for n in range(self.min_mode, self.max_mode + 1)
        ]

        # Sample number of superimposed modes
        num_modes = int(rng.integers(self.min_active_modes, self.max_active_modes + 1))

        selected_indices = rng.choice(len(all_modes), size=num_modes, replace=False)
        selected_modes = [all_modes[i] for i in selected_indices]

        u_raw = np.zeros(grid.shape, dtype=np.float64)
        mode_params: list[dict[str, Any]] = []

        lx = float(grid.x[-1])
        ly = float(grid.y[-1])

        for m, n in selected_modes:
            amp = float(rng.uniform(-1.0, 1.0))
            mode_field = np.sin(m * np.pi * grid.X / lx) * np.sin(
                n * np.pi * grid.Y / ly
            )
            u_raw += amp * mode_field
            mode_params.append({"m": int(m), "n": int(n), "amplitude": amp})

        # Normalize peak amplitude to target range for uniform dataset scale
        max_abs = float(np.max(np.abs(u_raw)))
        if max_abs > 1e-12:
            target_peak = float(rng.uniform(*self.target_peak_range))
            u_normalized = (u_raw / max_abs) * target_peak
        else:
            u_normalized = u_raw
            target_peak = 0.0

        # Ensure exact zero boundaries
        u_final = self.bc.apply(u_normalized)

        metadata: dict[str, Any] = {
            "generator": "fourier",
            "modes": mode_params,
            "num_active_modes": num_modes,
            "target_peak": target_peak,
        }
        return u_final, metadata


class GaussianSourceGenerator(InitialConditionGenerator):
    """Generates localized Gaussian thermal peaks with smooth boundary tapering.

    The continuous formulation is:
        u_raw(x, y) = sum_k A_k * exp(- ((x - x_k)^2 + (y - y_k)^2) / (2 * sigma_k^2))
        u0(x, y)    = u_raw(x, y) * W(x, y)

    where W(x, y) = sin(pi * x / Lx) * sin(pi * y / Ly) is a smooth C-infinity Dirichlet
    envelope window that identically zeroes the Gaussian tails at the plate boundaries.

    Attributes:
        min_sources: Minimum number of Gaussian spots (default: 1).
        max_sources: Maximum number of Gaussian spots (default: 3).
        center_margin: Minimum normalized distance from boundary edges (default: 0.15).
        sigma_range: Min and max Gaussian standard deviations as fraction of plate size.
        amplitude_range: Range of initial peak thermal amplitudes.
    """

    def __init__(
        self,
        min_sources: int = 1,
        max_sources: int = 3,
        center_margin: float = 0.15,
        sigma_range: tuple[float, float] = (0.05, 0.15),
        amplitude_range: tuple[float, float] = (0.5, 1.0),
    ) -> None:
        if min_sources < 1 or max_sources < min_sources:
            raise ValueError(
                f"Invalid source count: min={min_sources}, max={max_sources}"
            )
        if not (0.0 < center_margin < 0.5):
            raise ValueError(f"center_margin must be in (0, 0.5), got {center_margin}")
        if sigma_range[0] <= 0 or sigma_range[1] < sigma_range[0]:
            raise ValueError(f"Invalid sigma_range: {sigma_range}")

        self.min_sources = min_sources
        self.max_sources = max_sources
        self.center_margin = center_margin
        self.sigma_range = sigma_range
        self.amplitude_range = amplitude_range
        self.bc = ZeroDirichletBoundary()

    def generate(
        self,
        grid: Grid2D,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Generate randomized Gaussian thermal peaks with zero Dirichlet envelope."""
        num_sources = int(rng.integers(self.min_sources, self.max_sources + 1))
        lx = float(grid.x[-1])
        ly = float(grid.y[-1])
        min_dim = min(lx, ly)

        u_raw = np.zeros(grid.shape, dtype=np.float64)
        sources_params: list[dict[str, Any]] = []

        for _ in range(num_sources):
            # Place centers away from outer boundary margins
            cx = float(
                rng.uniform(self.center_margin * lx, (1.0 - self.center_margin) * lx)
            )
            cy = float(
                rng.uniform(self.center_margin * ly, (1.0 - self.center_margin) * ly)
            )
            sigma = float(
                rng.uniform(
                    self.sigma_range[0] * min_dim, self.sigma_range[1] * min_dim
                )
            )
            amp = float(rng.uniform(*self.amplitude_range))

            dist_sq = (grid.X - cx) ** 2 + (grid.Y - cy) ** 2
            gaussian = amp * np.exp(-dist_sq / (2.0 * (sigma**2)))
            u_raw += gaussian

            sources_params.append(
                {
                    "center_x": cx,
                    "center_y": cy,
                    "sigma": sigma,
                    "amplitude": amp,
                }
            )

        # Apply smooth C-infinity Dirichlet boundary window W(x, y)
        window = np.sin(np.pi * grid.X / lx) * np.sin(np.pi * grid.Y / ly)
        u_tapered = u_raw * window

        # Rescale peak to preserve target amplitude scale
        max_val = float(np.max(u_tapered))
        target_amp = float(rng.uniform(*self.amplitude_range))
        if max_val > 1e-12:
            u_scaled = (u_tapered / max_val) * target_amp
        else:
            u_scaled = u_tapered

        # Explicitly enforce boundary clamp to guarantee exact 0.0 values
        u_final = self.bc.apply(u_scaled)

        metadata: dict[str, Any] = {
            "generator": "gaussian",
            "num_sources": num_sources,
            "sources": sources_params,
            "target_amplitude": target_amp,
        }
        return u_final, metadata
