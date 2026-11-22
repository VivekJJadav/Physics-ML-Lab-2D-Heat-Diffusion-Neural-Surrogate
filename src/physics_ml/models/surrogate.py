"""Neural network surrogate architectures for 2D heat equation modeling.

This module implements a compact 2D convolutional residual network (HeatResNet2D)
designed to learn the one-step temporal operator u(x, y, t) -> u(x, y, t + dt).
"""

import torch
from torch import nn


class ConvBlock2D(nn.Module):
    """Two-layer 2D convolutional residual block with GELU activations."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Residual block forward pass: x + Conv2(GELU(Conv1(x)))."""
        residual = self.conv2(self.act1(self.conv1(x)))
        return self.act2(x + residual)


class HeatResNet2D(nn.Module):
    """Compact 2D Residual Convolutional Surrogate for heat diffusion.

    The network predicts the discrete finite-time increment:
        u(t + dt) = u(t) + F_theta(u(t))
    and enforces exact zero Dirichlet boundary conditions as a hard constraint.

    Attributes:
        in_channels: Number of input channels (default: 1 for temperature field).
        hidden_channels: Feature map channels across residual blocks (default: 32).
        num_blocks: Number of stacked ConvBlock2D blocks (default: 3).
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 32,
        num_blocks: int = 3,
        enforce_zero_boundary: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_blocks = num_blocks
        self.enforce_zero_boundary = enforce_zero_boundary

        # Spatial feature extraction
        self.in_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # Deep residual processing
        self.blocks = nn.ModuleList(
            [ConvBlock2D(hidden_channels) for _ in range(num_blocks)]
        )

        # Output projection predicting thermal increment delta_u
        self.out_proj = nn.Conv2d(
            hidden_channels, in_channels, kernel_size=3, padding=1
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Forward pass predicting u(t + dt) from current temperature field u(t).

        Args:
            u: Input temperature tensor of shape (B, 1, H, W).

        Returns:
            Predicted temperature tensor at t + dt of shape (B, 1, H, W).
        """
        if u.ndim != 4 or u.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected 4D input of shape (B, {self.in_channels}, H, W), "
                f"got {u.shape}"
            )

        feat = self.in_proj(u)
        for block in self.blocks:
            feat = block(feat)
        delta_u = self.out_proj(feat)

        # Global residual connection: u(t + dt) = u(t) + delta_u
        u_next = u + delta_u

        # Enforce exact zero Dirichlet boundary condition as a hard constraint
        if self.enforce_zero_boundary:
            u_next = self._apply_boundary_clamping(u_next)

        return u_next

    @staticmethod
    def _apply_boundary_clamping(field: torch.Tensor) -> torch.Tensor:
        """Clamp outer perimeter nodes to 0.0 to satisfy zero Dirichlet BCs."""
        # Clone to avoid in-place gradient mutation issues
        clamped = field.clone()
        clamped[:, :, 0, :] = 0.0
        clamped[:, :, -1, :] = 0.0
        clamped[:, :, :, 0] = 0.0
        clamped[:, :, :, -1] = 0.0
        return clamped

    def count_parameters(self) -> int:
        """Return the total number of trainable model parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
