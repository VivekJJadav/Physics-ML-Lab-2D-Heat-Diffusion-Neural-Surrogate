"""Tests for dataset generation, IC synthesis, splitting, and serialization.

Verifies Milestone 3 requirements:
- Fourier-mode generator respects m,n in {1,2,3,4} and zero Dirichlet boundaries.
- Gaussian-source generator respects parameter bounds and preserves zero boundaries.
- SimulationDataPipeline runs solver and records trajectories with correct shapes.
- All numerical arrays contain finite values (no NaNs, no Infs).
- Splitting enforces zero overlap between train, val, and test simulation IDs.
- Deterministic reproducibility under identical random seeds.
- Disk serialization roundtrip preserves all numerical values and metadata.
- PyTorch Dataset adapter formats tensors correctly for surrogate model consumption.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from physics_ml.config import SimulationConfig
from physics_ml.data import (
    FourierModeGenerator,
    GaussianSourceGenerator,
    HeatDiffusionDataset,
    SimulationDataPipeline,
    SimulationTrajectory,
    load_dataset,
    save_dataset,
)
from physics_ml.simulation.grid import Grid2D


@pytest.fixture
def test_config() -> SimulationConfig:
    """Compact simulation configuration for fast testing."""
    return SimulationConfig(
        plate_width=1.0,
        plate_height=1.0,
        grid_resolution_x=21,
        grid_resolution_y=21,
        thermal_diffusivity=0.01,
        simulation_duration=0.1,
        time_step=0.005,
    )


def test_fourier_mode_generator_boundaries_and_modes(
    test_config: SimulationConfig,
) -> None:
    """Verify FourierModeGenerator preserves zero boundaries and samples valid modes."""
    grid = Grid2D.from_config(test_config)
    gen = FourierModeGenerator(min_mode=1, max_mode=4)
    rng = np.random.default_rng(123)

    for _ in range(5):
        u0, meta = gen.generate(grid, rng)

        # Shape check
        assert u0.shape == (21, 21)

        # Finite values check
        assert np.all(np.isfinite(u0))

        # Peak amplitude within configured range
        assert 0.5 <= np.max(np.abs(u0)) <= 1.0 + 1e-6

        # Exact zero Dirichlet boundary check
        assert np.all(u0[0, :] == 0.0)
        assert np.all(u0[-1, :] == 0.0)
        assert np.all(u0[:, 0] == 0.0)
        assert np.all(u0[:, -1] == 0.0)

        # Metadata mode indices check (m, n in {1, 2, 3, 4})
        assert meta["generator"] == "fourier"
        for mode in meta["modes"]:
            assert 1 <= mode["m"] <= 4
            assert 1 <= mode["n"] <= 4


def test_gaussian_source_generator_boundaries_and_parameters(
    test_config: SimulationConfig,
) -> None:
    """Verify GaussianSourceGenerator preserves zero boundaries and parameter bounds."""
    grid = Grid2D.from_config(test_config)
    gen = GaussianSourceGenerator(min_sources=1, max_sources=3, center_margin=0.15)
    rng = np.random.default_rng(456)

    for _ in range(5):
        u0, meta = gen.generate(grid, rng)

        assert u0.shape == (21, 21)
        assert np.all(np.isfinite(u0))

        # Boundaries must be identically 0.0
        assert np.all(u0[0, :] == 0.0)
        assert np.all(u0[-1, :] == 0.0)
        assert np.all(u0[:, 0] == 0.0)
        assert np.all(u0[:, -1] == 0.0)

        # Centers must be inside interior margins
        assert meta["generator"] == "gaussian"
        for s in meta["sources"]:
            assert 0.15 <= s["center_x"] <= 0.85
            assert 0.15 <= s["center_y"] <= 0.85
            assert s["sigma"] > 0.0
            assert s["amplitude"] > 0.0


def test_generator_reproducibility_with_fixed_seed(
    test_config: SimulationConfig,
) -> None:
    """Verify identical outputs when generators are called with identical RNG seeds."""
    grid = Grid2D.from_config(test_config)
    fourier_gen = FourierModeGenerator()
    gaussian_gen = GaussianSourceGenerator()

    # Fourier reproducibility
    u0_a, meta_a = fourier_gen.generate(grid, np.random.default_rng(999))
    u0_b, meta_b = fourier_gen.generate(grid, np.random.default_rng(999))
    assert np.array_equal(u0_a, u0_b)
    assert meta_a == meta_b

    # Gaussian reproducibility
    u0_c, meta_c = gaussian_gen.generate(grid, np.random.default_rng(999))
    u0_d, meta_d = gaussian_gen.generate(grid, np.random.default_rng(999))
    assert np.array_equal(u0_c, u0_d)
    assert meta_c == meta_d


def test_simulation_pipeline_generation(test_config: SimulationConfig) -> None:
    """Verify end-to-end dataset generation from the pipeline."""
    pipeline = SimulationDataPipeline(
        config=test_config,
        num_snapshots=5,
    )
    splits = pipeline.generate_dataset(
        num_simulations=10,
        fourier_ratio=0.5,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=101,
    )

    assert splits.total_simulations == 10
    assert len(splits.train) == 7
    assert len(splits.val) == 2
    assert len(splits.test) == 1

    # Check snapshots shape across trajectories
    for traj in splits.train + splits.val + splits.test:
        assert traj.snapshots.shape == (5, 21, 21)
        assert len(traj.timestamps) == 5
        assert np.all(np.isfinite(traj.snapshots))
        # Zero Dirichlet boundary preservation on all snapshots
        assert np.all(traj.snapshots[:, 0, :] == 0.0)
        assert np.all(traj.snapshots[:, -1, :] == 0.0)
        assert np.all(traj.snapshots[:, :, 0] == 0.0)
        assert np.all(traj.snapshots[:, :, -1] == 0.0)


def test_dataset_splitting_isolation_no_overlap(
    test_config: SimulationConfig,
) -> None:
    """Verify that no simulation ID or initial condition is shared between splits."""
    pipeline = SimulationDataPipeline(config=test_config, num_snapshots=3)
    splits = pipeline.generate_dataset(num_simulations=20, seed=202)

    train_ids = set(splits.train_ids)
    val_ids = set(splits.val_ids)
    test_ids = set(splits.test_ids)

    # Disjoint sets check
    assert len(train_ids & val_ids) == 0, "Train and Val share simulation IDs"
    assert len(train_ids & test_ids) == 0, "Train and Test share simulation IDs"
    assert len(val_ids & test_ids) == 0, "Val and Test share simulation IDs"
    assert len(train_ids | val_ids | test_ids) == 20


def test_dataset_serialization_roundtrip(test_config: SimulationConfig) -> None:
    """Verify saving and loading dataset splits preserves exact arrays and metadata."""
    pipeline = SimulationDataPipeline(config=test_config, num_snapshots=4)
    original_splits = pipeline.generate_dataset(num_simulations=10, seed=303)

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = save_dataset(original_splits, Path(tmp_dir) / "test_dataset")
        loaded_splits = load_dataset(save_path)

        assert loaded_splits.total_simulations == original_splits.total_simulations
        assert loaded_splits.train_ids == original_splits.train_ids
        assert loaded_splits.val_ids == original_splits.val_ids
        assert loaded_splits.test_ids == original_splits.test_ids

        # Verify numerical snapshots match with machine precision
        for orig, loaded in zip(
            original_splits.train, loaded_splits.train, strict=True
        ):
            assert orig.simulation_id == loaded.simulation_id
            assert orig.generator_type == loaded.generator_type
            assert np.array_equal(orig.timestamps, loaded.timestamps)
            assert np.allclose(orig.snapshots, loaded.snapshots, atol=1e-12)


def test_serialization_validation_mismatched_timestamps(
    test_config: SimulationConfig,
) -> None:
    """Verify save_dataset raises ValueError on mismatched trajectory timestamps."""
    pipeline = SimulationDataPipeline(config=test_config, num_snapshots=4)
    splits = pipeline.generate_dataset(num_simulations=4, seed=304)

    # Invalidate timestamps of one trajectory
    bad_traj = splits.train[0]
    mismatched_traj = SimulationTrajectory(
        simulation_id=bad_traj.simulation_id,
        generator_type=bad_traj.generator_type,
        ic_params=bad_traj.ic_params,
        timestamps=np.array([0.0, 0.5, 1.0, 1.5]),
        snapshots=bad_traj.snapshots,
        config=bad_traj.config,
    )
    splits.train[0] = mismatched_traj

    with tempfile.TemporaryDirectory() as tmp_dir:
        with pytest.raises(ValueError, match="Timestamp mismatch"):
            save_dataset(splits, Path(tmp_dir) / "invalid_dataset")


def test_pytorch_dataset_step_mode(test_config: SimulationConfig) -> None:
    """Verify PyTorch Dataset adapter in one-step Markovian mode."""
    pipeline = SimulationDataPipeline(config=test_config, num_snapshots=5)
    splits = pipeline.generate_dataset(num_simulations=4, seed=404)

    # 4 simulations * (5 - 1) step pairs = 16 training samples
    torch_ds = HeatDiffusionDataset(splits.train, mode="step")
    expected_samples = len(splits.train) * (5 - 1)
    assert len(torch_ds) == expected_samples

    sample = torch_ds[0]
    assert "input" in sample
    assert "target" in sample
    assert "dt" in sample
    assert "time" in sample
    assert "sim_id" in sample

    # Check PyTorch tensor types and shapes: (C, H, W) = (1, 21, 21)
    u_in = sample["input"]
    u_out = sample["target"]
    assert isinstance(u_in, torch.Tensor)
    assert isinstance(u_out, torch.Tensor)
    assert u_in.shape == (1, 21, 21)
    assert u_out.shape == (1, 21, 21)
    assert u_in.dtype == torch.float32
    assert u_out.dtype == torch.float32
    assert torch.all(torch.isfinite(u_in))
    assert torch.all(torch.isfinite(u_out))


def test_pytorch_dataset_time_conditioned_mode(
    test_config: SimulationConfig,
) -> None:
    """Verify PyTorch Dataset adapter in time-conditioned mode."""
    pipeline = SimulationDataPipeline(config=test_config, num_snapshots=6)
    splits = pipeline.generate_dataset(num_simulations=3, seed=505)

    # In time_conditioned mode, u0 -> u(t_k) for k in 1..5
    torch_ds = HeatDiffusionDataset(splits.train, mode="time_conditioned")
    expected_samples = len(splits.train) * (6 - 1)
    assert len(torch_ds) == expected_samples

    sample = torch_ds[0]
    assert sample["input"].shape == (1, 21, 21)
    assert sample["target"].shape == (1, 21, 21)
    assert float(sample["time"]) > 0.0
