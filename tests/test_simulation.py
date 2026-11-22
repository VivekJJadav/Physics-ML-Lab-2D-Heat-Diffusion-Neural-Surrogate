"""Tests for 2D FTCS numerical solver, boundary conditions, and analytical verification.

Verifies Milestone 2 requirements:
- Grid2D spatial coordinate mapping and spacing.
- Zero Dirichlet boundary condition preservation across time steps.
- Preservation of initial conditions at t = 0.
- Physical temperature evolution (dissipation and maximum principle).
- Enforcement of stability condition (rx + ry <= 0.5) and rejection of unstable setups.
- Agreement with single-mode analytical benchmark at multiple time instances.
- Spatial grid convergence under refined discretization.
"""

import numpy as np
import pytest

from physics_ml.config import SimulationConfig
from physics_ml.evaluation import compute_l2_error, compute_max_error, compute_rmse
from physics_ml.physics import (
    analytical_decay_rate,
    single_mode_analytical_solution,
)
from physics_ml.simulation import (
    FTCSSolver2D,
    Grid2D,
    NumericalInstabilityError,
)


def test_grid2d_initialization() -> None:
    """Verify that Grid2D coordinates, spacings, and shapes match the config."""
    config = SimulationConfig(
        plate_width=2.0,
        plate_height=1.0,
        grid_resolution_x=21,
        grid_resolution_y=11,
    )
    grid = Grid2D.from_config(config)

    assert grid.shape == (21, 11)
    assert grid.x.shape == (21,)
    assert grid.y.shape == (11,)
    assert grid.X.shape == (21, 11)
    assert grid.Y.shape == (21, 11)
    assert pytest.approx(grid.dx, rel=1e-6) == 2.0 / 20.0
    assert pytest.approx(grid.dy, rel=1e-6) == 1.0 / 10.0
    assert grid.x[0] == 0.0
    assert grid.x[-1] == 2.0
    assert grid.y[0] == 0.0
    assert grid.y[-1] == 1.0


def test_boundary_preservation() -> None:
    """Verify that zero Dirichlet boundaries remain 0.0 after multiple solver steps."""
    config = SimulationConfig(
        grid_resolution_x=25,
        grid_resolution_y=25,
        thermal_diffusivity=0.01,
        time_step=0.005,
        simulation_duration=0.1,
    )
    solver = FTCSSolver2D(config)
    grid = solver.grid

    # Create a non-zero interior field with zero boundaries
    u = np.sin(np.pi * grid.X) * np.sin(np.pi * grid.Y)
    u_curr = u.copy()

    for _ in range(15):
        u_curr = solver.step(u_curr)
        # Check all four boundaries
        assert np.all(u_curr[0, :] == 0.0)
        assert np.all(u_curr[-1, :] == 0.0)
        assert np.all(u_curr[:, 0] == 0.0)
        assert np.all(u_curr[:, -1] == 0.0)


def test_initial_conditions() -> None:
    """Verify solver correctly captures and preserves initial condition at t = 0."""
    config = SimulationConfig(
        grid_resolution_x=21,
        grid_resolution_y=21,
        time_step=0.005,
        simulation_duration=0.05,
    )
    solver = FTCSSolver2D(config)
    grid = solver.grid

    u0 = np.sin(np.pi * grid.X) * np.sin(np.pi * grid.Y)
    result = solver.solve(u0)

    assert result.times[0] == 0.0
    assert np.allclose(result.states[0], u0)
    assert len(result.times) == config.num_time_steps + 1
    assert result.final_time == pytest.approx(config.simulation_duration)


def test_initial_condition_shape_mismatch() -> None:
    """Verify that passing an array with incorrect dimensions raises ValueError."""
    config = SimulationConfig(grid_resolution_x=21, grid_resolution_y=21)
    solver = FTCSSolver2D(config)
    wrong_u0 = np.zeros((10, 10))

    with pytest.raises(ValueError, match="!= grid shape"):
        solver.solve(wrong_u0)


def test_temperature_evolution_and_dissipation() -> None:
    """Verify energy and peak temperature decay under zero Dirichlet BCs."""
    config = SimulationConfig(
        grid_resolution_x=31,
        grid_resolution_y=31,
        thermal_diffusivity=0.01,
        time_step=0.002,
        simulation_duration=0.1,
    )
    solver = FTCSSolver2D(config)
    grid = solver.grid

    # Hot spot at plate center
    u0 = np.exp(-((grid.X - 0.5) ** 2 + (grid.Y - 0.5) ** 2) / 0.02)
    solver.boundary_condition.apply(u0)

    result = solver.solve(u0, record_interval=5)

    max_temps = [np.max(state) for state in result.states]
    total_energies = [np.sum(state) for state in result.states]

    # Monotonic decrease in peak temperature and integrated energy
    for i in range(len(max_temps) - 1):
        assert max_temps[i + 1] <= max_temps[i]
        assert total_energies[i + 1] <= total_energies[i]

    # Discrete maximum principle: temperatures remain within [0, max(u0)]
    for state in result.states:
        assert np.all(state >= -1e-12)
        assert np.all(state <= np.max(u0) + 1e-12)


def test_stability_validation_accepts_stable_config() -> None:
    """Verify that configurations satisfying rx + ry <= 0.5 are accepted."""
    # rx + ry = 0.25 <= 0.5
    stable_config = SimulationConfig(
        plate_width=1.0,
        plate_height=1.0,
        grid_resolution_x=51,
        grid_resolution_y=51,
        thermal_diffusivity=0.01,
        time_step=0.005,
    )
    solver = FTCSSolver2D(stable_config)
    assert solver.total_r <= 0.5


def test_stability_validation_rejects_unstable_config() -> None:
    """Verify that unstable configs (rx + ry > 0.5) raise NumericalInstabilityError."""
    # dx = dy = 0.02. If dt = 0.015, rx = ry = 0.375 => rx + ry = 0.75 > 0.5
    unstable_config = SimulationConfig(
        plate_width=1.0,
        plate_height=1.0,
        grid_resolution_x=51,
        grid_resolution_y=51,
        thermal_diffusivity=0.01,
        time_step=0.015,
    )

    with pytest.raises(NumericalInstabilityError) as exc_info:
        FTCSSolver2D(unstable_config)

    err = exc_info.value
    assert err.total_r > 0.5
    assert "2D FTCS numerical instability detected" in str(err)
    assert f"dt_max={unstable_config.max_stable_time_step:.6e}" in str(err)


@pytest.mark.parametrize("eval_time", [0.05, 0.1, 0.2, 0.5, 1.0])
def test_analytical_agreement_at_multiple_times(eval_time: float) -> None:
    """Verify high-accuracy agreement between FTCS solver and analytical benchmark."""
    config = SimulationConfig(
        plate_width=1.0,
        plate_height=1.0,
        grid_resolution_x=51,
        grid_resolution_y=51,
        thermal_diffusivity=0.01,
        time_step=0.001,
        simulation_duration=eval_time,
    )
    solver = FTCSSolver2D(config)
    grid = solver.grid

    # Exact initial condition u(x, y, 0) = sin(pi*x) * sin(pi*y)
    u0 = single_mode_analytical_solution(grid.X, grid.Y, 0.0, config)
    result = solver.solve(u0)

    u_num = result.final_state
    u_exact = single_mode_analytical_solution(grid.X, grid.Y, eval_time, config)

    rel_l2 = compute_l2_error(u_num, u_exact, relative=True)
    max_err = compute_max_error(u_num, u_exact)
    rmse = compute_rmse(u_num, u_exact)

    # Relative L2 error should be under 0.5% for this fine discretization
    assert rel_l2 < 0.005, f"Rel L2 error {rel_l2:.6f} exceeded 0.005 at t={eval_time}"
    assert max_err < 0.005, f"Max error {max_err:.6f} exceeded 0.005 at t={eval_time}"
    assert rmse < 0.003, f"RMSE {rmse:.6f} exceeded 0.003 at t={eval_time}"


def test_spatial_grid_convergence() -> None:
    """Verify that numerical error decreases with spatial refinement at O(dx^2) rate."""
    resolutions = [21, 41, 81]
    errors: list[float] = []
    dx_values: list[float] = []

    eval_time = 0.1
    alpha = 0.01

    # Keep Fourier number rx = ry = 0.125 constant across resolutions
    # rx = alpha * dt / dx^2 = 0.125 => dt = 0.125 * dx^2 / alpha
    for n in resolutions:
        dx = 1.0 / (n - 1)
        dt = (0.125 * (dx**2)) / alpha

        config = SimulationConfig(
            plate_width=1.0,
            plate_height=1.0,
            grid_resolution_x=n,
            grid_resolution_y=n,
            thermal_diffusivity=alpha,
            time_step=dt,
            simulation_duration=eval_time,
        )
        solver = FTCSSolver2D(config)
        grid = solver.grid

        u0 = single_mode_analytical_solution(grid.X, grid.Y, 0.0, config)
        result = solver.solve(u0)

        u_exact = single_mode_analytical_solution(
            grid.X, grid.Y, result.final_time, config
        )
        l2_err = compute_l2_error(result.final_state, u_exact, relative=False)

        errors.append(l2_err)
        dx_values.append(dx)

    # 1. Monotonic decrease in error as grid resolution increases
    assert errors[0] > errors[1] > errors[2], (
        f"Errors did not decrease monotonically: {errors}"
    )

    # 2. Estimate empirical convergence order p = log(E1 / E2) / log(dx1 / dx2)
    order_1 = np.log(errors[0] / errors[1]) / np.log(dx_values[0] / dx_values[1])
    order_2 = np.log(errors[1] / errors[2]) / np.log(dx_values[1] / dx_values[2])

    # Theoretical spatial order for central difference with dt ~ dx^2 is 2.0
    assert 1.8 <= order_1 <= 2.2, f"Convergence order {order_1:.2f} != ~2.0"
    assert 1.8 <= order_2 <= 2.2, f"Convergence order {order_2:.2f} != ~2.0"


def test_analytical_solution_validation() -> None:
    """Verify analytical solution boundary values and parameter constraints."""
    config = SimulationConfig()
    grid = Grid2D.from_config(config)

    # Negative evaluation time raises ValueError
    with pytest.raises(ValueError, match="must be non-negative"):
        single_mode_analytical_solution(grid.X, grid.Y, -0.1, config)

    # Analytical solution must be zero on boundaries at all times
    u_exact = single_mode_analytical_solution(grid.X, grid.Y, 0.5, config)
    assert np.allclose(u_exact[0, :], 0.0)
    assert np.allclose(u_exact[-1, :], 0.0)
    assert np.allclose(u_exact[:, 0], 0.0)
    assert np.allclose(u_exact[:, -1], 0.0)

    # Analytical decay rate matches continuous formula
    expected_decay = (
        config.thermal_diffusivity
        * (np.pi**2)
        * ((1.0 / (config.plate_width**2)) + (1.0 / (config.plate_height**2)))
    )
    assert pytest.approx(analytical_decay_rate(config)) == expected_decay
