"""Physical diagnostic tools for verifying physical invariants and failure modes.

Tracks:
1. Total thermal energy and L2 energy dissipation monotonicity.
2. Discrete maximum principle adherence (spurious heating / undershoot).
3. 2D FFT radial power spectrum for spectral bias and high-frequency noise.
4. Out-of-Distribution (OOD) stress testing across unseen frequencies and gradients.
"""

from typing import Any

import numpy as np
import torch
from torch import nn

from physics_ml.config import SimulationConfig
from physics_ml.data.generators import FourierModeGenerator, GaussianSourceGenerator
from physics_ml.evaluation.metrics import (
    compute_l2_error,
    compute_max_error,
)
from physics_ml.evaluation.rollout import RolloutEvaluator
from physics_ml.simulation.ftcs import FTCSSolver2D
from physics_ml.simulation.grid import Grid2D


def compute_total_energy(u: np.ndarray, dx: float, dy: float) -> float:
    r"""Compute the total thermal energy integral over the spatial domain.

    E(t) = \iint_\Omega u(x, y, t) dx dy
         \approx \sum_{i,j} u_{i,j} \Delta x \Delta y

    Args:
        u: 2D temperature field of shape (Nx, Ny).
        dx: Spatial grid spacing along x.
        dy: Spatial grid spacing along y.

    Returns:
        Scalar integrated thermal energy.
    """
    return float(np.sum(u) * dx * dy)


def compute_l2_energy(u: np.ndarray, dx: float, dy: float) -> float:
    r"""Compute the L2 energy functional (Lyapunov functional) over the domain.

    H(t) = \iint_\Omega u^2(x, y, t) dx dy
         \approx \sum_{i,j} u_{i,j}^2 \Delta x \Delta y

    For the heat equation with homogeneous Dirichlet boundary conditions,
    dH/dt = -\alpha \int_\Omega ||\nabla u||^2 <= 0 (strictly non-increasing).

    Args:
        u: 2D temperature field of shape (Nx, Ny).
        dx: Spatial grid spacing along x.
        dy: Spatial grid spacing along y.

    Returns:
        Scalar integrated L2 squared energy.
    """
    return float(np.sum(u**2) * dx * dy)


def compute_energy_trajectory(
    snapshots: np.ndarray,
    dx: float,
    dy: float,
) -> dict[str, np.ndarray]:
    """Compute total energy and L2 energy histories along a trajectory.

    Args:
        snapshots: 3D array of shape (T, Nx, Ny).
        dx: Spatial grid spacing along x.
        dy: Spatial grid spacing along y.

    Returns:
        Dictionary with arrays 'total_energy' and 'l2_energy' of length T.
    """
    total_e = [compute_total_energy(s, dx, dy) for s in snapshots]
    l2_e = [compute_l2_energy(s, dx, dy) for s in snapshots]
    return {
        "total_energy": np.array(total_e),
        "l2_energy": np.array(l2_e),
    }


def check_energy_monotonicity(
    energy_history: np.ndarray,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Verify whether energy monotonically dissipates over time.

    In the heat equation with zero Dirichlet boundary and non-negative field,
    energy should never increase from one step to the next:
    E(t_{k+1}) <= E(t_k) + tolerance

    Args:
        energy_history: 1D array of energy values over sequential time steps.
        tolerance: Permissible numerical noise threshold.

    Returns:
        Dictionary summarizing violations, maximum increase, and pass flag.
    """
    diffs = np.diff(energy_history)
    violations = diffs > tolerance
    violation_indices = np.where(violations)[0] + 1

    max_increase = float(np.max(diffs)) if len(diffs) > 0 else 0.0

    return {
        "is_monotonic": len(violation_indices) == 0,
        "num_violations": len(violation_indices),
        "violation_step_indices": violation_indices.tolist(),
        "max_step_increase": max_increase,
        "total_energy_drift": float(energy_history[-1] - energy_history[0]),
    }


def check_maximum_principle(
    snapshots: np.ndarray,
    tolerance: float = 1e-5,
) -> dict[str, Any]:
    """Check adherence to the continuous/discrete maximum and minimum principles.

    For parabolic diffusion with homogeneous Dirichlet boundary conditions:
    1. Global bounds: min(0, min(u0)) <= u(x, y, t) <= max(0, max(u0)).
    2. Step-wise maximum monotonicity: max(u_{k+1}) <= max(u_k) + tolerance.
    3. Step-wise minimum monotonicity: min(u_{k+1}) >= min(u_k) - tolerance.

    Args:
        snapshots: 3D array of shape (T, Nx, Ny).
        tolerance: Permissible tolerance for small numerical/network imprecision.

    Returns:
        Dictionary containing violation counts, extreme values, and diagnostic flags.
    """
    u0 = snapshots[0]
    global_min_bound = min(0.0, float(np.min(u0)))
    global_max_bound = max(0.0, float(np.max(u0)))

    step_maxs = np.max(snapshots, axis=(1, 2))
    step_mins = np.min(snapshots, axis=(1, 2))

    # Check global bounds
    global_max_violation = float(np.max(step_maxs) - global_max_bound)
    global_min_violation = float(global_min_bound - np.min(step_mins))

    # Check step-wise monotonicity
    max_diffs = np.diff(step_maxs)
    min_diffs = np.diff(step_mins)

    spurious_heating_steps = np.where(max_diffs > tolerance)[0] + 1
    spurious_cooling_steps = np.where(min_diffs < -tolerance)[0] + 1

    is_valid = (
        global_max_violation <= tolerance
        and global_min_violation <= tolerance
        and len(spurious_heating_steps) == 0
    )

    return {
        "is_adherent": is_valid,
        "global_min_bound": global_min_bound,
        "global_max_bound": global_max_bound,
        "trajectory_min": float(np.min(step_mins)),
        "trajectory_max": float(np.max(step_maxs)),
        "global_max_violation": max(0.0, global_max_violation),
        "global_min_violation": max(0.0, global_min_violation),
        "spurious_heating_steps": spurious_heating_steps.tolist(),
        "spurious_cooling_steps": spurious_cooling_steps.tolist(),
        "step_maxima": step_maxs.tolist(),
        "step_minima": step_mins.tolist(),
    }


def compute_radial_power_spectrum(
    u: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the azimuthally averaged radial power spectrum of a 2D scalar field.

    Uses 2D Fast Fourier Transform (FFT) to quantify spatial frequency content.

    Args:
        u: 2D temperature field of shape (Nx, Ny).

    Returns:
        Tuple of (radial_frequencies, averaged_power_density).
    """
    nx, ny = u.shape
    f_shift = np.fft.fftshift(np.fft.fft2(u))
    power_2d = np.abs(f_shift) ** 2 / (nx * ny)

    # Coordinate grids centered at zero frequency
    kx = np.fft.fftshift(np.fft.fftfreq(nx)) * nx
    ky = np.fft.fftshift(np.fft.fftfreq(ny)) * ny
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")

    # Radial wavenumber
    k_radial = np.sqrt(kx_grid**2 + ky_grid**2)
    max_k = int(np.floor(min(nx, ny) / 2))

    # Bin into integer radial shells
    k_bins = np.arange(0, max_k + 1)
    power_1d = np.zeros(max_k)

    for i in range(max_k):
        mask = (k_radial >= k_bins[i]) & (k_radial < k_bins[i + 1])
        if np.any(mask):
            power_1d[i] = np.mean(power_2d[mask])
        else:
            power_1d[i] = 0.0

    radial_coords = 0.5 * (k_bins[:-1] + k_bins[1:])
    return radial_coords, power_1d


def run_ood_stress_tests(
    model: nn.Module,
    config: SimulationConfig,
    device: str | torch.device = "cpu",
    seed: int = 1234,
) -> dict[str, Any]:
    """Evaluate surrogate robustness against Out-of-Distribution (OOD) initial states.

    Stress scenarios:
    1. High-frequency Fourier modes: m, n in {5, 6} (unseen spatial frequencies).
    2. Narrow Gaussian source: sigma = 0.02 (steeper gradients than training).
    3. High amplitude Gaussian: A = 2.5 (2.5x larger than training max ~1.0).

    Args:
        model: Trained surrogate neural network.
        config: SimulationConfig specifying domain and solver grid.
        device: Device for neural execution.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary of comparative performance across OOD stress regimes.
    """
    grid = Grid2D.from_config(config)
    solver = FTCSSolver2D(config)
    evaluator = RolloutEvaluator(model, device=device)

    snapshot_interval = 0.1
    record_interval = max(1, round(snapshot_interval / config.time_step))
    num_rollout_steps = round(config.simulation_duration / snapshot_interval)

    results: dict[str, Any] = {}

    # Scenario 1: High-Frequency Fourier Modes (m, n in {5, 6})
    rng1 = np.random.default_rng(seed)
    fourier_gen = FourierModeGenerator(
        min_mode=5,
        max_mode=6,
        min_active_modes=2,
        max_active_modes=3,
        target_peak_range=(0.5, 0.8),
    )
    u0_hf, _ = fourier_gen.generate(grid, rng1)
    gt_hf = solver.solve(u0_hf, record_interval=record_interval).states
    pred_hf = evaluator.predict_rollout(u0_hf, num_steps=num_rollout_steps)

    hf_rel_l2 = [
        compute_l2_error(pred_hf[k], gt_hf[k], relative=True) for k in range(len(gt_hf))
    ]
    hf_max_err = [compute_max_error(pred_hf[k], gt_hf[k]) for k in range(len(gt_hf))]
    results["high_frequency"] = {
        "description": "High-frequency Fourier modes (m,n in {5,6})",
        "1step_rel_l2": float(hf_rel_l2[1]),
        "final_rel_l2": float(hf_rel_l2[-1]),
        "final_max_error": float(hf_max_err[-1]),
        "rel_l2_history": hf_rel_l2,
        "max_principle": check_maximum_principle(pred_hf),
    }

    # Scenario 2: Narrow Gaussian Source (sigma = 0.02 - sharp gradient)
    rng2 = np.random.default_rng(seed + 1)
    gauss_sharp_gen = GaussianSourceGenerator(
        min_sources=1,
        max_sources=1,
        center_margin=0.2,
        sigma_range=(0.02, 0.02),
        amplitude_range=(0.8, 1.0),
    )
    u0_sharp, _ = gauss_sharp_gen.generate(grid, rng2)
    gt_sharp = solver.solve(u0_sharp, record_interval=record_interval).states
    pred_sharp = evaluator.predict_rollout(u0_sharp, num_steps=num_rollout_steps)

    sharp_rel_l2 = [
        compute_l2_error(pred_sharp[k], gt_sharp[k], relative=True)
        for k in range(len(gt_sharp))
    ]
    sharp_max_err = [
        compute_max_error(pred_sharp[k], gt_sharp[k]) for k in range(len(gt_sharp))
    ]
    results["sharp_gradient"] = {
        "description": "Sharp localized Gaussian (sigma=0.02)",
        "1step_rel_l2": float(sharp_rel_l2[1]),
        "final_rel_l2": float(sharp_rel_l2[-1]),
        "final_max_error": float(sharp_max_err[-1]),
        "rel_l2_history": sharp_rel_l2,
        "max_principle": check_maximum_principle(pred_sharp),
    }

    # Scenario 3: High Amplitude Gaussian (A = 2.5)
    rng3 = np.random.default_rng(seed + 2)
    gauss_high_amp_gen = GaussianSourceGenerator(
        min_sources=1,
        max_sources=2,
        center_margin=0.2,
        sigma_range=(0.08, 0.12),
        amplitude_range=(2.5, 2.5),
    )
    u0_high, _ = gauss_high_amp_gen.generate(grid, rng3)
    gt_high = solver.solve(u0_high, record_interval=record_interval).states
    pred_high = evaluator.predict_rollout(u0_high, num_steps=num_rollout_steps)

    high_rel_l2 = [
        compute_l2_error(pred_high[k], gt_high[k], relative=True)
        for k in range(len(gt_high))
    ]
    high_max_err = [
        compute_max_error(pred_high[k], gt_high[k]) for k in range(len(gt_high))
    ]
    results["high_amplitude"] = {
        "description": "High amplitude Gaussian (A=2.5)",
        "1step_rel_l2": float(high_rel_l2[1]),
        "final_rel_l2": float(high_rel_l2[-1]),
        "final_max_error": float(high_max_err[-1]),
        "rel_l2_history": high_rel_l2,
        "max_principle": check_maximum_principle(pred_high),
    }

    return results
