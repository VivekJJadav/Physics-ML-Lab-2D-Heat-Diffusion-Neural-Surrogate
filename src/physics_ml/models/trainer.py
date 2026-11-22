"""Training and evaluation harness for neural surrogate models.

Supports:
- Standard Mean Squared Error (MSE) loss on temperature fields.
- Physics-informed Sobolev Laplacian regularization:
    L_total = L_MSE + lambda * L_Laplacian
  where L_Laplacian penalizes discrepancies in the discrete central-difference
  spatial Laplacian on interior nodes.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def compute_discrete_laplacian_torch(
    u: torch.Tensor,
    dx: float,
    dy: float,
) -> torch.Tensor:
    """Compute central difference discrete Laplacian on interior nodes [1:-1, 1:-1].

    Excludes the outermost boundary where values are clamped.

    Args:
        u: 4D tensor of shape (batch, channels, Nx, Ny).
        dx: Spatial grid spacing in x direction.
        dy: Spatial grid spacing in y direction.

    Returns:
        Laplacian tensor of shape (batch, channels, Nx - 2, Ny - 2).
    """
    lap_x = (u[:, :, 2:, 1:-1] - 2.0 * u[:, :, 1:-1, 1:-1] + u[:, :, :-2, 1:-1]) / (
        dx**2
    )
    lap_y = (u[:, :, 1:-1, 2:] - 2.0 * u[:, :, 1:-1, 1:-1] + u[:, :, 1:-1, :-2]) / (
        dy**2
    )
    return lap_x + lap_y


def compute_laplacian_reference_scale(
    targets: torch.Tensor,
    dx: float,
    dy: float,
) -> float:
    """Calculate fixed normalization scale S_lap_squared = mean((Laplacian(u_ref))^2).

    Strictly computed across training target snapshots on interior nodes.

    Args:
        targets: 4D tensor of shape (N, 1, Nx, Ny) containing reference target fields.
        dx: Spatial grid spacing in x direction.
        dy: Spatial grid spacing in y direction.

    Returns:
        Scalar S_lap_squared = mean((nabla_h^2 u_ref)^2) over interior nodes.
    """
    lap = compute_discrete_laplacian_torch(targets, dx, dy)
    return float(torch.mean(lap**2).item())


@dataclass
class TrainingHistory:
    """Records epoch-by-epoch loss metrics and checkpoint provenance."""

    train_losses: list[float]
    val_losses: list[float]
    best_epoch: int
    best_val_loss: float
    train_mse_losses: list[float] | None = None
    val_mse_losses: list[float] | None = None
    train_lap_losses: list[float] | None = None
    val_lap_losses: list[float] | None = None
    train_lap_norm_losses: list[float] | None = None
    val_lap_norm_losses: list[float] | None = None
    weighted_val_lap_losses: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert history into a dictionary format."""
        d: dict[str, Any] = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
        }
        if self.train_mse_losses is not None:
            d["train_mse_losses"] = self.train_mse_losses
            d["val_mse_losses"] = self.val_mse_losses
            d["train_lap_losses"] = self.train_lap_losses
            d["val_lap_losses"] = self.val_lap_losses
        if self.train_lap_norm_losses is not None:
            d["train_lap_norm_losses"] = self.train_lap_norm_losses
            d["val_lap_norm_losses"] = self.val_lap_norm_losses
        if self.weighted_val_lap_losses is not None:
            d["weighted_val_lap_losses"] = self.weighted_val_lap_losses
        return d


class SurrogateTrainer:
    """Orchestrates model training, validation tracking, and checkpoint saving."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader[dict[str, torch.Tensor | str]],
        val_loader: DataLoader[dict[str, torch.Tensor | str]],
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        lambda_laplacian: float = 0.0,
        laplacian_scale: float | None = None,
        epsilon: float = 1e-8,
        dx: float = 0.02,
        dy: float = 0.02,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lambda_laplacian = float(lambda_laplacian)
        self.laplacian_scale = (
            float(laplacian_scale) if laplacian_scale is not None else None
        )
        self.epsilon = float(epsilon)
        self.dx = float(dx)
        self.dy = float(dy)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.last_epoch_train_mse = 0.0
        self.last_epoch_train_lap = 0.0
        self.last_epoch_train_lap_norm = 0.0

    def train_epoch(self) -> float:
        """Execute one complete training epoch over the training dataloader.

        Returns:
            Scalar average total loss (MSE + lambda * Normalized_Laplacian_MSE).
        """
        self.model.train()
        total_loss = 0.0
        total_mse = 0.0
        total_lap = 0.0
        total_lap_norm = 0.0
        num_batches = 0

        for batch in self.train_loader:
            u_in = batch["input"].to(self.device)
            u_target = batch["target"].to(self.device)

            self.optimizer.zero_grad()
            u_pred = self.model(u_in)
            mse_loss = self.criterion(u_pred, u_target)

            if self.lambda_laplacian > 0.0:
                lap_pred = compute_discrete_laplacian_torch(u_pred, self.dx, self.dy)
                lap_target = compute_discrete_laplacian_torch(
                    u_target, self.dx, self.dy
                )
                lap_raw_loss = self.criterion(lap_pred, lap_target)
                if self.laplacian_scale is not None and self.laplacian_scale > 0.0:
                    lap_loss = lap_raw_loss / (self.laplacian_scale + self.epsilon)
                else:
                    lap_loss = lap_raw_loss

                loss = mse_loss + self.lambda_laplacian * lap_loss
                total_lap += float(lap_raw_loss.item())
                total_lap_norm += float(lap_loss.item())
            else:
                loss = mse_loss

            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item())
            total_mse += float(mse_loss.item())
            num_batches += 1

        n = max(1, num_batches)
        self.last_epoch_train_mse = total_mse / n
        self.last_epoch_train_lap = total_lap / n
        self.last_epoch_train_lap_norm = total_lap_norm / n
        return total_loss / n

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader[dict[str, torch.Tensor | str]],
    ) -> dict[str, float]:
        """Compute MSE, RMSE, relative L2, and Laplacian error metrics across a loader.

        Returns:
            Dictionary containing both field-level and Laplacian error metrics.
        """
        self.model.eval()
        total_mse = 0.0
        total_l2_diff = 0.0
        total_l2_ref = 0.0
        total_lap_mse = 0.0
        total_lap_diff = 0.0
        total_lap_ref = 0.0
        num_batches = 0

        for batch in loader:
            u_in = batch["input"].to(self.device)
            u_target = batch["target"].to(self.device)

            u_pred = self.model(u_in)
            mse = self.criterion(u_pred, u_target)
            total_mse += float(mse.item())

            diff = u_pred - u_target
            total_l2_diff += float(torch.sum(diff**2).item())
            total_l2_ref += float(torch.sum(u_target**2).item())

            # Interior discrete Laplacian
            lap_pred = compute_discrete_laplacian_torch(u_pred, self.dx, self.dy)
            lap_target = compute_discrete_laplacian_torch(u_target, self.dx, self.dy)
            lap_mse = self.criterion(lap_pred, lap_target)
            total_lap_mse += float(lap_mse.item())

            lap_diff = lap_pred - lap_target
            total_lap_diff += float(torch.sum(lap_diff**2).item())
            total_lap_ref += float(torch.sum(lap_target**2).item())

            num_batches += 1

        n = max(1, num_batches)
        mean_mse = total_mse / n
        rmse = float(np.sqrt(mean_mse))
        rel_l2 = (
            float(np.sqrt(total_l2_diff / total_l2_ref))
            if total_l2_ref > 1e-12
            else 0.0
        )

        mean_lap_mse = total_lap_mse / n
        lap_rmse = float(np.sqrt(mean_lap_mse))
        lap_rel_l2 = (
            float(np.sqrt(total_lap_diff / total_lap_ref))
            if total_lap_ref > 1e-12
            else 0.0
        )

        if self.laplacian_scale is not None and self.laplacian_scale > 0.0:
            norm_lap_mse = mean_lap_mse / (self.laplacian_scale + self.epsilon)
        else:
            norm_lap_mse = mean_lap_mse

        weighted_lap = self.lambda_laplacian * norm_lap_mse
        total_loss = mean_mse + weighted_lap

        return {
            "total_loss": total_loss,
            "mse": mean_mse,
            "rmse": rmse,
            "relative_l2": rel_l2,
            "laplacian_mse": mean_lap_mse,
            "laplacian_norm_mse": norm_lap_mse,
            "laplacian_rmse": lap_rmse,
            "laplacian_relative_l2": lap_rel_l2,
            "weighted_laplacian": weighted_lap,
        }

    def fit(
        self,
        epochs: int,
        checkpoint_path: Path | str | None = None,
        scheduler: Any | None = None,
    ) -> TrainingHistory:
        """Train model for specified epochs, checkpointing best validation model.

        Args:
            epochs: Total training epochs.
            checkpoint_path: Optional path to save the best model weights.
            scheduler: Optional learning rate scheduler.

        Returns:
            TrainingHistory instance with epoch records.
        """
        train_losses: list[float] = []
        val_losses: list[float] = []
        train_mse_losses: list[float] = []
        val_mse_losses: list[float] = []
        train_lap_losses: list[float] = []
        val_lap_losses: list[float] = []
        train_lap_norm_losses: list[float] = []
        val_lap_norm_losses: list[float] = []
        weighted_val_lap_losses: list[float] = []

        best_val_loss = float("inf")
        best_epoch = -1

        save_dest = Path(checkpoint_path) if checkpoint_path else None
        if save_dest:
            save_dest.parent.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, epochs + 1):
            train_total = self.train_epoch()
            val_metrics = self.evaluate(self.val_loader)
            val_total = val_metrics["total_loss"]

            train_losses.append(train_total)
            val_losses.append(val_total)
            train_mse_losses.append(self.last_epoch_train_mse)
            val_mse_losses.append(val_metrics["mse"])
            train_lap_losses.append(self.last_epoch_train_lap)
            val_lap_losses.append(val_metrics["laplacian_mse"])
            train_lap_norm_losses.append(self.last_epoch_train_lap_norm)
            val_lap_norm_losses.append(val_metrics["laplacian_norm_mse"])
            weighted_val_lap_losses.append(val_metrics["weighted_laplacian"])

            if scheduler is not None:
                scheduler.step()

            # Checkpoint best model based on validation total loss
            if val_total < best_val_loss:
                best_val_loss = val_total
                best_epoch = epoch
                if save_dest:
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "val_loss": val_total,
                            "val_mse": val_metrics["mse"],
                            "val_lap_mse": val_metrics["laplacian_mse"],
                            "val_lap_norm_mse": val_metrics["laplacian_norm_mse"],
                            "lambda_laplacian": self.lambda_laplacian,
                            "laplacian_scale": self.laplacian_scale,
                        },
                        save_dest,
                    )

        return TrainingHistory(
            train_losses=train_losses,
            val_losses=val_losses,
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            train_mse_losses=train_mse_losses,
            val_mse_losses=val_mse_losses,
            train_lap_losses=train_lap_losses,
            val_lap_losses=val_lap_losses,
            train_lap_norm_losses=train_lap_norm_losses,
            val_lap_norm_losses=val_lap_norm_losses,
            weighted_val_lap_losses=weighted_val_lap_losses,
        )

    def load_checkpoint(self, checkpoint_path: Path | str) -> None:
        """Load model weights from a saved checkpoint file."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
