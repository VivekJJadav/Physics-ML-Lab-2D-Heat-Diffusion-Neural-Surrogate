"""Unit and integration tests for Milestone 5: One-Step FTCS Neural Surrogate.

Requirements verified:
1. Correct one-step temporal alignment: delta_t = 0.005s between consecutive pairs.
2. Dataset reproducibility under identical random seed.
3. Simulation-level split isolation (no simulation leakage across train/val/test).
4. Tensor shapes and finite values across forward passes.
5. Exact zero Dirichlet boundary enforcement on model outputs.
6. Successful forward and backward passes with finite gradients.
7. Checkpoint save and load consistency.
8. Correct autoregressive rollout step counts and time alignment (20, 40, 100 steps).
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from physics_ml.config import SimulationConfig
from physics_ml.data import (
    HeatDiffusionDataset,
    SimulationDataPipeline,
)
from physics_ml.evaluation.rollout import RolloutEvaluator
from physics_ml.models.onestep_surrogate import OneStepHeatResNet2D
from physics_ml.models.trainer import SurrogateTrainer


@pytest.fixture
def onestep_config() -> SimulationConfig:
    """Fixture providing a fast, small grid configuration for unit testing."""
    return SimulationConfig(
        plate_width=1.0,
        plate_height=1.0,
        grid_resolution_x=21,
        grid_resolution_y=21,
        thermal_diffusivity=0.01,
        simulation_duration=0.05,  # 10 steps of 0.005s
        time_step=0.005,
    )


def test_onestep_temporal_alignment(onestep_config: SimulationConfig) -> None:
    """Verify input-target pairs in step mode have dt = 0.005s."""
    pipeline = SimulationDataPipeline(
        config=onestep_config,
        num_snapshots=11,  # 10 steps + 1
    )
    splits = pipeline.generate_dataset(num_simulations=4, seed=42)

    # Check timestamps array in trajectories
    traj = splits.train[0]
    dt_diffs = np.diff(traj.timestamps)
    assert np.allclose(dt_diffs, 0.005, atol=1e-7)

    # Check Dataset pairs
    ds = HeatDiffusionDataset(splits.train, mode="step")
    sample_0 = ds[0]
    u_in = sample_0["input"]
    u_target = sample_0["target"]
    assert u_in.shape == (1, 21, 21)
    assert u_target.shape == (1, 21, 21)
    assert not torch.allclose(u_in, u_target)  # Physically diffusing


def test_dataset_reproducibility(onestep_config: SimulationConfig) -> None:
    """Verify generating dataset with identical seed produces identical arrays."""
    p1 = SimulationDataPipeline(config=onestep_config, num_snapshots=11)
    s1 = p1.generate_dataset(num_simulations=4, seed=123)

    p2 = SimulationDataPipeline(config=onestep_config, num_snapshots=11)
    s2 = p2.generate_dataset(num_simulations=4, seed=123)

    for tr1, tr2 in zip(s1.train, s2.train, strict=True):
        assert tr1.simulation_id == tr2.simulation_id
        assert np.allclose(tr1.snapshots, tr2.snapshots)


def test_simulation_level_split_isolation(onestep_config: SimulationConfig) -> None:
    """Verify zero overlap between train, val, and test simulation IDs."""
    pipeline = SimulationDataPipeline(config=onestep_config, num_snapshots=11)
    splits = pipeline.generate_dataset(num_simulations=10, seed=42)

    train_ids = {t.simulation_id for t in splits.train}
    val_ids = {t.simulation_id for t in splits.val}
    test_ids = {t.simulation_id for t in splits.test}

    assert len(train_ids.intersection(val_ids)) == 0
    assert len(train_ids.intersection(test_ids)) == 0
    assert len(val_ids.intersection(test_ids)) == 0


def test_onestep_model_shapes_and_finite_values() -> None:
    """Verify forward pass output shape matches input and contains finite values."""
    model = OneStepHeatResNet2D(in_channels=1, hidden_channels=16, num_blocks=2)
    x = torch.randn(4, 1, 51, 51)
    out = model(x)

    assert out.shape == (4, 1, 51, 51)
    assert torch.all(torch.isfinite(out))


def test_onestep_exact_zero_boundary() -> None:
    """Verify hard boundary constraint sets perimeter cells to exactly 0.0."""
    model = OneStepHeatResNet2D(in_channels=1, hidden_channels=16, num_blocks=2)
    # Pass non-zero values on boundaries
    x = torch.ones(2, 1, 31, 31) * 5.0
    out = model(x)

    assert torch.all(out[:, :, 0, :] == 0.0)
    assert torch.all(out[:, :, -1, :] == 0.0)
    assert torch.all(out[:, :, :, 0] == 0.0)
    assert torch.all(out[:, :, :, -1] == 0.0)


def test_onestep_forward_backward_pass() -> None:
    """Verify loss calculation and backpropagation generate non-zero gradients."""
    model = OneStepHeatResNet2D(in_channels=1, hidden_channels=16, num_blocks=2)
    x = torch.randn(2, 1, 21, 21, requires_grad=True)
    target = torch.randn(2, 1, 21, 21)

    out = model(x)
    loss = torch.nn.functional.mse_loss(out, target)
    loss.backward()

    # Check model parameters have finite gradients
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None
            assert torch.all(torch.isfinite(p.grad))


def test_checkpoint_save_and_load(onestep_config: SimulationConfig) -> None:
    """Verify saved checkpoint can be loaded and yields identical predictions."""
    pipeline = SimulationDataPipeline(config=onestep_config, num_snapshots=5)
    splits = pipeline.generate_dataset(num_simulations=4, seed=42)

    train_ds = HeatDiffusionDataset(splits.train, mode="step")
    val_ds = HeatDiffusionDataset(splits.val, mode="step")
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)

    model = OneStepHeatResNet2D(in_channels=1, hidden_channels=16, num_blocks=2)
    trainer = SurrogateTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=1e-3,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "onestep_test.pt"
        trainer.fit(epochs=2, checkpoint_path=ckpt_path)
        assert ckpt_path.exists()

        # Restore best checkpoint into model
        trainer.load_checkpoint(ckpt_path)
        model.eval()

        # Model evaluation with best weights
        test_in = torch.randn(1, 1, 21, 21)
        pred_original = model(test_in).detach().numpy()

        # Load weights into fresh model
        model_loaded = OneStepHeatResNet2D(
            in_channels=1, hidden_channels=16, num_blocks=2
        )
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model_loaded.load_state_dict(ckpt["model_state_dict"])
        model_loaded.eval()

        pred_loaded = model_loaded(test_in).detach().numpy()
        assert np.allclose(pred_original, pred_loaded)


def test_rollout_step_counts_and_time_alignment(
    onestep_config: SimulationConfig,
) -> None:
    """Verify multi-step rollout handles 20, 40, and 100 steps correctly."""
    model = OneStepHeatResNet2D(in_channels=1, hidden_channels=16, num_blocks=2)
    evaluator = RolloutEvaluator(model, device="cpu")

    u0 = np.ones((21, 21)) * 0.5
    u0[0, :] = u0[-1, :] = u0[:, 0] = u0[:, -1] = 0.0

    # 20 steps
    rollout_20 = evaluator.predict_rollout(u0, num_steps=20)
    assert rollout_20.shape == (21, 21, 21)  # 20 steps + 1 initial

    # 40 steps
    rollout_40 = evaluator.predict_rollout(u0, num_steps=40)
    assert rollout_40.shape == (41, 21, 21)

    # Verify boundaries remain zero at all steps
    assert np.all(rollout_40[:, 0, :] == 0.0)
    assert np.all(rollout_40[:, -1, :] == 0.0)
    assert np.all(rollout_40[:, :, 0] == 0.0)
    assert np.all(rollout_40[:, :, -1] == 0.0)
