"""Tests for HeatResNet2D neural surrogate architecture and training harness.

Verifies Milestone 4 requirements:
- Model forward pass accepts 4D tensors and preserves spatial dimensions.
- Hard zero Dirichlet boundary clamping layer guarantees exact boundary zeros.
- Trainable parameter count is compact (< 100,000 parameters).
- Trainer execution records losses and saves best validation checkpoints.
"""

import tempfile
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from physics_ml.config import SimulationConfig
from physics_ml.data import HeatDiffusionDataset, SimulationDataPipeline
from physics_ml.models import HeatResNet2D, SurrogateTrainer


def test_model_forward_pass_and_shape() -> None:
    """Verify forward pass preserves spatial shape and outputs finite values."""
    model = HeatResNet2D(in_channels=1, hidden_channels=16, num_blocks=2)
    x = torch.randn(4, 1, 51, 51)
    y = model(x)

    assert y.shape == (4, 1, 51, 51)
    assert torch.all(torch.isfinite(y))


def test_model_hard_zero_boundary_enforcement() -> None:
    """Verify that model output boundary pixels are identically 0.0."""
    model = HeatResNet2D(in_channels=1, hidden_channels=16, num_blocks=2)
    # Pass non-zero input even at boundaries
    x = torch.ones(2, 1, 31, 31)
    y = model(x)

    assert torch.all(y[:, :, 0, :] == 0.0)
    assert torch.all(y[:, :, -1, :] == 0.0)
    assert torch.all(y[:, :, :, 0] == 0.0)
    assert torch.all(y[:, :, :, -1] == 0.0)


def test_model_compact_parameter_count() -> None:
    """Verify model parameter count stays compact to prevent overfitting."""
    model = HeatResNet2D(in_channels=1, hidden_channels=32, num_blocks=3)
    num_params = model.count_parameters()

    # Around 56,000 parameters - well below 100,000
    assert 20_000 < num_params < 100_000


def test_model_invalid_input_shape() -> None:
    """Verify passing tensor with incorrect dimensionality raises ValueError."""
    model = HeatResNet2D(in_channels=1)
    with pytest.raises(ValueError, match="Expected 4D input"):
        model(torch.randn(1, 51, 51))


def test_surrogate_trainer_fit_and_checkpoint() -> None:
    """Verify end-to-end trainer execution, loss recording, and checkpoint saving."""
    config = SimulationConfig(
        plate_width=1.0,
        plate_height=1.0,
        grid_resolution_x=21,
        grid_resolution_y=21,
        thermal_diffusivity=0.01,
        simulation_duration=0.1,
        time_step=0.005,
    )
    pipeline = SimulationDataPipeline(config=config, num_snapshots=4)
    splits = pipeline.generate_dataset(num_simulations=4, seed=42)

    train_ds = HeatDiffusionDataset(splits.train, mode="step")
    val_ds = HeatDiffusionDataset(splits.val, mode="step")

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    model = HeatResNet2D(in_channels=1, hidden_channels=16, num_blocks=2)
    trainer = SurrogateTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=1e-3,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "test_model.pt"
        history = trainer.fit(epochs=2, checkpoint_path=ckpt_path)

        assert len(history.train_losses) == 2
        assert len(history.val_losses) == 2
        assert history.best_epoch in (1, 2)
        assert history.best_val_loss > 0.0
        assert ckpt_path.exists()

        # Test loading checkpoint
        trainer.load_checkpoint(ckpt_path)
        eval_metrics = trainer.evaluate(val_loader)
        assert eval_metrics["mse"] > 0.0
        assert eval_metrics["rmse"] > 0.0


def test_compute_discrete_laplacian_torch_accuracy() -> None:
    """Verify PyTorch discrete Laplacian calculation against analytical function."""
    from physics_ml.models.trainer import compute_discrete_laplacian_torch

    nx, ny = 21, 21
    dx, dy = 1.0 / (nx - 1), 1.0 / (ny - 1)
    x = torch.linspace(0, 1, nx)
    y = torch.linspace(0, 1, ny)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    u = (torch.sin(torch.pi * xx) * torch.sin(torch.pi * yy)).unsqueeze(0).unsqueeze(0)

    lap = compute_discrete_laplacian_torch(u, dx, dy)
    assert lap.shape == (1, 1, nx - 2, ny - 2)

    # Analytical Laplacian is -2 * pi^2 * u
    u_interior = u[:, :, 1:-1, 1:-1]
    expected_lap = -2.0 * (torch.pi**2) * u_interior
    # Second-order central difference error O(dx^2)
    assert torch.allclose(lap, expected_lap, atol=0.05, rtol=0.05)


def test_surrogate_trainer_with_laplacian_loss() -> None:
    """Verify trainer execution and checkpointing with lambda_laplacian > 0."""
    config = SimulationConfig(
        grid_resolution_x=21,
        grid_resolution_y=21,
        simulation_duration=0.1,
        time_step=0.005,
    )
    pipeline = SimulationDataPipeline(config=config, num_snapshots=4)
    splits = pipeline.generate_dataset(num_simulations=4, seed=42)

    train_ds = HeatDiffusionDataset(splits.train, mode="step")
    val_ds = HeatDiffusionDataset(splits.val, mode="step")

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    model = HeatResNet2D(in_channels=1, hidden_channels=16, num_blocks=2)
    trainer = SurrogateTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=1e-3,
        lambda_laplacian=0.01,
        dx=config.dx,
        dy=config.dy,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "test_lap_model.pt"
        history = trainer.fit(epochs=2, checkpoint_path=ckpt_path)

        assert len(history.train_losses) == 2
        assert len(history.val_losses) == 2
        assert history.train_lap_losses is not None
        assert len(history.train_lap_losses) == 2
        assert ckpt_path.exists()

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        assert checkpoint["lambda_laplacian"] == 0.01
        assert "val_lap_mse" in checkpoint


def test_surrogate_trainer_normalized_laplacian_scale() -> None:
    """Verify trainer execution with normalized Laplacian scale S_lap_squared."""
    from physics_ml.models.trainer import compute_laplacian_reference_scale

    config = SimulationConfig(
        grid_resolution_x=21,
        grid_resolution_y=21,
        simulation_duration=0.1,
        time_step=0.005,
    )
    pipeline = SimulationDataPipeline(config=config, num_snapshots=4)
    splits = pipeline.generate_dataset(num_simulations=4, seed=42)

    import numpy as np

    # Collect reference target tensors from training split
    train_targets = []
    for traj in splits.train:
        for t in range(len(traj.timestamps) - 1):
            train_targets.append(traj.snapshots[t + 1])
    target_tensor = torch.from_numpy(np.array(train_targets)).unsqueeze(1).float()

    s_lap_sq = compute_laplacian_reference_scale(target_tensor, config.dx, config.dy)
    assert s_lap_sq > 0.0

    train_ds = HeatDiffusionDataset(splits.train, mode="step")
    val_ds = HeatDiffusionDataset(splits.val, mode="step")

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    model = HeatResNet2D(in_channels=1, hidden_channels=16, num_blocks=2)
    trainer = SurrogateTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=1e-3,
        lambda_laplacian=0.05,
        laplacian_scale=s_lap_sq,
        dx=config.dx,
        dy=config.dy,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "test_norm_lap_model.pt"
        history = trainer.fit(epochs=2, checkpoint_path=ckpt_path)

        assert len(history.train_losses) == 2
        assert history.train_lap_norm_losses is not None
        assert len(history.train_lap_norm_losses) == 2
        assert history.val_lap_norm_losses is not None
        assert len(history.val_lap_norm_losses) == 2

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        assert checkpoint["lambda_laplacian"] == 0.05
        assert checkpoint["laplacian_scale"] == s_lap_sq
        assert "val_lap_norm_mse" in checkpoint
