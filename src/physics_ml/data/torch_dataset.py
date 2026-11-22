"""PyTorch Dataset adapter for consuming heat diffusion simulation snapshots."""

from typing import Any, Literal

import torch
from torch.utils.data import Dataset

from physics_ml.data.dataset import SimulationTrajectory


class HeatDiffusionDataset(Dataset[dict[str, torch.Tensor | str]]):
    """PyTorch Dataset bridging simulation trajectories to training loops.

    Supports two primary scientific ML learning paradigms:
    1. 'step': One-step Markovian autoregressive prediction:
       Input:  u(x, y, t_k)       of shape (1, Nx, Ny)
       Target: u(x, y, t_{k+1})   of shape (1, Nx, Ny)
       Used for continuous rollouts, neural ODEs, or residual operators.

    2. 'time_conditioned': Direct temporal solution mapping from initial state:
       Input:  u_0(x, y)          of shape (1, Nx, Ny)
               t (elapsed time)   scalar float
       Target: u(x, y, t)         of shape (1, Nx, Ny)
       Used for Fourier Neural Operators (FNO), DeepONets, or encoder-decoders.

    Attributes:
        trajectories: List of SimulationTrajectory instances.
        mode: Learning paradigm ('step' or 'time_conditioned').
    """

    def __init__(
        self,
        trajectories: list[SimulationTrajectory],
        mode: Literal["step", "time_conditioned"] = "step",
    ) -> None:
        if mode not in ("step", "time_conditioned"):
            raise ValueError(
                f"Unsupported mode: '{mode}'. Choose 'step' or 'time_conditioned'"
            )

        self.trajectories = trajectories
        self.mode = mode
        self._samples: list[dict[str, Any]] = []
        self._build_index()

    def _build_index(self) -> None:
        """Flatten trajectories into individual (input, target) sample indices."""
        for traj in self.trajectories:
            num_snaps = traj.num_snapshots
            if self.mode == "step":
                # Create (t_k, t_{k+1}) consecutive step pairs
                for k in range(num_snaps - 1):
                    self._samples.append(
                        {
                            "traj": traj,
                            "u_in": traj.snapshots[k],
                            "u_out": traj.snapshots[k + 1],
                            "t_in": float(traj.timestamps[k]),
                            "t_out": float(traj.timestamps[k + 1]),
                            "dt": float(traj.timestamps[k + 1] - traj.timestamps[k]),
                            "sim_id": traj.simulation_id,
                        }
                    )
            elif self.mode == "time_conditioned":
                # Create (u0, t_k) -> u(t_k) pairs for all k >= 1
                u0 = traj.initial_state
                for k in range(1, num_snaps):
                    self._samples.append(
                        {
                            "traj": traj,
                            "u_in": u0,
                            "u_out": traj.snapshots[k],
                            "t_in": 0.0,
                            "t_out": float(traj.timestamps[k]),
                            "dt": float(traj.timestamps[k]),
                            "sim_id": traj.simulation_id,
                        }
                    )

    def __len__(self) -> int:
        """Return total number of training pairs across all trajectories."""
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        """Fetch a single training pair formatted as PyTorch tensors.

        Returns:
            Dictionary containing:
                - 'input': FloatTensor of shape (1, Nx, Ny)
                - 'target': FloatTensor of shape (1, Nx, Ny)
                - 'dt': FloatTensor scalar
                - 'time': FloatTensor scalar evaluation time
                - 'sim_id': String simulation provenance identifier
        """
        sample = self._samples[idx]

        # Add channel dimension (C, H, W) = (1, Nx, Ny) standard for 2D convolutions
        u_in_tensor = torch.from_numpy(sample["u_in"]).unsqueeze(0).to(torch.float32)
        u_out_tensor = torch.from_numpy(sample["u_out"]).unsqueeze(0).to(torch.float32)

        return {
            "input": u_in_tensor,
            "target": u_out_tensor,
            "dt": torch.tensor(sample["dt"], dtype=torch.float32),
            "time": torch.tensor(sample["t_out"], dtype=torch.float32),
            "sim_id": sample["sim_id"],
        }
