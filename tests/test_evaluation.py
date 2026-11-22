"""Tests for evaluation metrics, multi-step rollout, and physical diagnostics."""

import numpy as np
import pytest
import torch
from torch import nn

from physics_ml.config import SimulationConfig
from physics_ml.data.dataset import SimulationTrajectory
from physics_ml.evaluation.diagnostics import (
    check_energy_monotonicity,
    check_maximum_principle,
    compute_l2_energy,
    compute_radial_power_spectrum,
    compute_total_energy,
    run_ood_stress_tests,
)
from physics_ml.evaluation.rollout import RolloutEvaluator
from physics_ml.models.surrogate import HeatResNet2D


class DummyDecayModel(nn.Module):
    """Simple linear decay model for fast deterministic evaluation testing."""

    def __init__(self, decay_factor: float = 0.95) -> None:
        super().__init__()
        self.decay_factor = decay_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x * self.decay_factor
        # Enforce zero boundary
        out[:, :, 0, :] = 0.0
        out[:, :, -1, :] = 0.0
        out[:, :, :, 0] = 0.0
        out[:, :, :, -1] = 0.0
        return out


@pytest.fixture
def sample_config() -> SimulationConfig:
    return SimulationConfig(
        grid_resolution_x=21,
        grid_resolution_y=21,
        simulation_duration=0.5,
        time_step=0.005,
    )


@pytest.fixture
def dummy_trajectory(sample_config: SimulationConfig) -> SimulationTrajectory:
    # 5 snapshots, 21x21 grid
    times = np.linspace(0.0, 0.4, 5)
    snapshots = np.zeros((5, 21, 21), dtype=np.float64)
    x = np.linspace(0, 1, 21)
    y = np.linspace(0, 1, 21)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    mode = np.sin(np.pi * xx) * np.sin(np.pi * yy)
    for k, t in enumerate(times):
        snapshots[k] = mode * np.exp(-t)

    return SimulationTrajectory(
        simulation_id="test_sim_001",
        generator_type="fourier",
        ic_params={"modes": [1]},
        timestamps=times,
        snapshots=snapshots,
        config=sample_config,
    )


def test_predict_rollout_shape_and_initial_state() -> None:
    model = DummyDecayModel(0.9)
    evaluator = RolloutEvaluator(model)

    u0 = np.ones((21, 21), dtype=np.float64)
    rollout = evaluator.predict_rollout(u0, num_steps=4)

    assert rollout.shape == (5, 21, 21)
    assert np.allclose(rollout[0], u0)
    # Check boundaries clamped to zero after 1 step
    assert np.all(rollout[1, 0, :] == 0.0)
    assert np.all(rollout[1, -1, :] == 0.0)


def test_evaluate_trajectory_metrics(dummy_trajectory: SimulationTrajectory) -> None:
    model = DummyDecayModel(0.9)
    evaluator = RolloutEvaluator(model)

    res = evaluator.evaluate_trajectory(dummy_trajectory)

    assert res["simulation_id"] == "test_sim_001"
    assert len(res["relative_l2_errors"]) == 5
    assert len(res["max_errors"]) == 5
    # Step 0 error is exactly 0
    assert res["relative_l2_errors"][0] == 0.0
    assert res["max_errors"][0] == 0.0
    # Steps k > 0 have positive finite errors
    assert np.all(res["relative_l2_errors"][1:] > 0)
    assert np.all(np.isfinite(res["relative_l2_errors"]))


def test_evaluate_ensemble_statistics(dummy_trajectory: SimulationTrajectory) -> None:
    model = DummyDecayModel(0.9)
    evaluator = RolloutEvaluator(model)

    traj2 = SimulationTrajectory(
        simulation_id="test_sim_002",
        generator_type="gaussian",
        ic_params={"sigma": 0.1},
        timestamps=dummy_trajectory.timestamps,
        snapshots=dummy_trajectory.snapshots * 0.8,
        config=sample_config,
    )

    ensemble_res = evaluator.evaluate_ensemble([dummy_trajectory, traj2])

    assert ensemble_res["num_trajectories"] == 2
    assert len(ensemble_res["mean_relative_l2"]) == 5
    assert "fourier_subset" in ensemble_res
    assert "gaussian_subset" in ensemble_res
    assert ensemble_res["fourier_subset"]["count"] == 1
    assert ensemble_res["gaussian_subset"]["count"] == 1


def test_energy_computation_and_monotonicity() -> None:
    dx, dy = 0.05, 0.05
    u0 = np.ones((21, 21), dtype=np.float64)
    e_tot = compute_total_energy(u0, dx, dy)
    e_l2 = compute_l2_energy(u0, dx, dy)

    expected = 21 * 21 * dx * dy
    assert np.isclose(e_tot, expected)
    assert np.isclose(e_l2, expected)

    # Monotonically decaying energy
    decaying_energy = np.array([10.0, 8.5, 7.1, 6.0, 5.2])
    res_mono = check_energy_monotonicity(decaying_energy)
    assert res_mono["is_monotonic"] is True
    assert res_mono["num_violations"] == 0

    # Non-monotonic energy (spurious heat injection at step 3)
    spurious_energy = np.array([10.0, 8.5, 9.2, 7.0])
    res_spur = check_energy_monotonicity(spurious_energy)
    assert res_spur["is_monotonic"] is False
    assert res_spur["num_violations"] == 1
    assert res_spur["violation_step_indices"] == [2]


def test_maximum_principle_verification() -> None:
    # 4 snapshots with decaying maximums and positive values
    valid_snapshots = np.zeros((4, 11, 11), dtype=np.float64)
    for k in range(4):
        valid_snapshots[k, 1:-1, 1:-1] = 1.0 / (k + 1)

    res_valid = check_maximum_principle(valid_snapshots)
    assert res_valid["is_adherent"] is True
    assert res_valid["global_max_violation"] == 0.0
    assert len(res_valid["spurious_heating_steps"]) == 0

    # Violate with spurious heating
    invalid_snapshots = valid_snapshots.copy()
    invalid_snapshots[2, 5, 5] = 2.5  # Exceeds max(u0) = 1.0
    res_invalid = check_maximum_principle(invalid_snapshots)
    assert res_invalid["is_adherent"] is False
    assert res_invalid["global_max_violation"] > 1.0
    assert 2 in res_invalid["spurious_heating_steps"]


def test_radial_power_spectrum_shape_and_values() -> None:
    u = np.random.randn(32, 32)
    radial_coords, power = compute_radial_power_spectrum(u)

    assert len(radial_coords) == 16
    assert len(power) == 16
    assert np.all(power >= 0.0)
    assert np.all(radial_coords > 0.0)


def test_ood_stress_testing_harness(sample_config: SimulationConfig) -> None:
    model = HeatResNet2D(in_channels=1, hidden_channels=8, num_blocks=1)
    ood_results = run_ood_stress_tests(model, sample_config, seed=42)

    assert "high_frequency" in ood_results
    assert "sharp_gradient" in ood_results
    assert "high_amplitude" in ood_results

    for _key, data in ood_results.items():
        assert "1step_rel_l2" in data
        assert "final_rel_l2" in data
        assert "max_principle" in data
        assert data["final_rel_l2"] >= 0.0
