"""Autoregressive multi-step rollout evaluator for neural surrogate models."""

from typing import Any

import numpy as np
import torch
from torch import nn

from physics_ml.data.dataset import SimulationTrajectory
from physics_ml.evaluation.metrics import (
    compute_l2_error,
    compute_max_error,
    compute_rmse,
)


class RolloutEvaluator:
    """Evaluates multi-step autoregressive rollouts against ground truth.

    Attributes:
        model: Trained PyTorch neural surrogate model.
        device: Execution device ('cpu' or 'cuda').
    """

    def __init__(self, model: nn.Module, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_rollout(
        self,
        u0: np.ndarray,
        num_steps: int,
    ) -> np.ndarray:
        """Execute autonomous multi-step autoregressive rollout from initial state u0.

        At each step k >= 1:
            u_hat_{k} = model(u_hat_{k-1})

        Args:
            u0: Initial temperature field of shape (Nx, Ny).
            num_steps: Number of forward autoregressive steps to take.

        Returns:
            3D array of rolled-out states of shape (num_steps + 1, Nx, Ny).
        """
        if u0.ndim != 2:
            raise ValueError(
                f"Expected 2D initial state (Nx, Ny), got shape {u0.shape}"
            )
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")

        curr_state = (
            torch.from_numpy(u0.copy())
            .unsqueeze(0)
            .unsqueeze(0)
            .to(torch.float32)
            .to(self.device)
        )  # (1, 1, Nx, Ny)

        rollout_states = [u0.copy()]

        for _ in range(num_steps):
            next_state = self.model(curr_state)
            state_np = next_state.squeeze(0).squeeze(0).cpu().numpy().astype(np.float64)
            rollout_states.append(state_np)
            curr_state = next_state

        return np.stack(rollout_states, axis=0)

    def evaluate_trajectory(
        self,
        trajectory: SimulationTrajectory,
    ) -> dict[str, Any]:
        """Perform full multi-step rollout and compare against ground truth trajectory.

        Args:
            trajectory: Ground-truth SimulationTrajectory with K recorded snapshots.

        Returns:
            Dictionary containing step-by-step errors, predicted states, and metadata.
        """
        ground_truth = trajectory.snapshots  # (K+1, Nx, Ny)
        num_steps = len(ground_truth) - 1
        predicted = self.predict_rollout(trajectory.initial_state, num_steps=num_steps)

        rel_l2_errors: list[float] = []
        abs_l2_errors: list[float] = []
        max_errors: list[float] = []
        rmse_errors: list[float] = []

        for k in range(len(ground_truth)):
            u_pred = predicted[k]
            u_true = ground_truth[k]

            # At t=0, prediction is identically u0, error is 0.0
            if k == 0:
                rel_l2_errors.append(0.0)
                abs_l2_errors.append(0.0)
                max_errors.append(0.0)
                rmse_errors.append(0.0)
                continue

            rel_l2 = compute_l2_error(u_pred, u_true, relative=True)
            abs_l2 = compute_l2_error(u_pred, u_true, relative=False)
            max_err = compute_max_error(u_pred, u_true)
            rmse = compute_rmse(u_pred, u_true)

            rel_l2_errors.append(rel_l2)
            abs_l2_errors.append(abs_l2)
            max_errors.append(max_err)
            rmse_errors.append(rmse)

        return {
            "simulation_id": trajectory.simulation_id,
            "generator_type": trajectory.generator_type,
            "timestamps": trajectory.timestamps,
            "ground_truth": ground_truth,
            "predicted": predicted,
            "relative_l2_errors": np.array(rel_l2_errors),
            "abs_l2_errors": np.array(abs_l2_errors),
            "max_errors": np.array(max_errors),
            "rmse_errors": np.array(rmse_errors),
            "final_relative_l2": rel_l2_errors[-1],
            "final_max_error": max_errors[-1],
        }

    def evaluate_ensemble(
        self,
        trajectories: list[SimulationTrajectory],
    ) -> dict[str, Any]:
        """Evaluate multi-step rollout across an ensemble of trajectories.

        Computes mean error growth curves, subset metrics, and error dispersion.
        """
        if not trajectories:
            raise ValueError("Trajectory ensemble cannot be empty.")

        results = [self.evaluate_trajectory(t) for t in trajectories]

        # Stack error curves: shape (N_sims, num_steps + 1)
        rel_l2_stack = np.stack([r["relative_l2_errors"] for r in results], axis=0)
        max_err_stack = np.stack([r["max_errors"] for r in results], axis=0)
        rmse_stack = np.stack([r["rmse_errors"] for r in results], axis=0)

        timestamps = results[0]["timestamps"]

        # Aggregate metrics
        mean_rel_l2 = np.mean(rel_l2_stack, axis=0)
        std_rel_l2 = np.std(rel_l2_stack, axis=0)
        mean_max_err = np.mean(max_err_stack, axis=0)
        mean_rmse = np.mean(rmse_stack, axis=0)

        # Separate metrics by initial condition type
        fourier_results = [r for r in results if r["generator_type"] == "fourier"]
        gaussian_results = [r for r in results if r["generator_type"] == "gaussian"]

        def _sub_stats(sub: list[dict[str, Any]]) -> dict[str, Any]:
            if not sub:
                return {}
            sub_rel = np.stack([r["relative_l2_errors"] for r in sub], axis=0)
            sub_max = np.stack([r["max_errors"] for r in sub], axis=0)
            return {
                "count": len(sub),
                "mean_relative_l2": np.mean(sub_rel, axis=0).tolist(),
                "std_relative_l2": np.std(sub_rel, axis=0).tolist(),
                "final_mean_relative_l2": float(np.mean(sub_rel[:, -1])),
                "mean_max_error": np.mean(sub_max, axis=0).tolist(),
                "final_mean_max_error": float(np.mean(sub_max[:, -1])),
            }

        return {
            "num_trajectories": len(trajectories),
            "timestamps": timestamps.tolist(),
            "mean_relative_l2": mean_rel_l2.tolist(),
            "std_relative_l2": std_rel_l2.tolist(),
            "final_mean_relative_l2": float(mean_rel_l2[-1]),
            "mean_max_error": mean_max_err.tolist(),
            "final_mean_max_error": float(mean_max_err[-1]),
            "mean_rmse": mean_rmse.tolist(),
            "fourier_subset": _sub_stats(fourier_results),
            "gaussian_subset": _sub_stats(gaussian_results),
            "individual_runs": results,
        }
