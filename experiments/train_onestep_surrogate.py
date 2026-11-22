"""Milestone 5: One-Step Neural Surrogate Training and Comprehensive Evaluation.

Objective:
Investigate whether a neural network can learn the individual explicit FTCS
numerical update:
    u(x, y, t) -> u(x, y, t + 0.005)
and evaluate error accumulation when rolled out autoregressively for 20 steps (0.1s),
40 steps (0.2s), and 100 steps (0.5s).

Preserves all existing Milestone 4 models, checkpoints, and datasets intact.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from physics_ml.data import HeatDiffusionDataset, load_dataset
from physics_ml.evaluation.metrics import (
    compute_l2_error,
    compute_max_error,
)
from physics_ml.evaluation.rollout import RolloutEvaluator
from physics_ml.models.onestep_surrogate import OneStepHeatResNet2D
from physics_ml.models.trainer import (
    SurrogateTrainer,
    compute_discrete_laplacian_torch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Milestone 5: One-Step Neural Surrogate Training and Rollout Audit"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="outputs/datasets/heat_diffusion_onestep_v1",
        help="Path to one-step dataset directory.",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="outputs/models",
        help="Directory to save model checkpoints.",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default="outputs/figures",
        help="Directory to save visualization figures.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
        help="Training epochs (default: 60).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Initial learning rate (default: 1e-3).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master random seed (default: 42).",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Ensure exact reproducibility across PyTorch and NumPy."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_one_step(
    model: torch.nn.Module,
    test_trajectories: list[Any],
    dx: float,
    dy: float,
) -> dict[str, Any]:
    """Evaluate one-step predictive accuracy across all held-out test snapshot pairs.

    Compares Neural Surrogate, Identity Baseline (no change), and FTCS reference.
    """
    model.eval()

    all_u_in: list[np.ndarray] = []
    all_u_ref: list[np.ndarray] = []
    all_u_pred: list[np.ndarray] = []
    all_types: list[str] = []

    with torch.no_grad():
        for traj in test_trajectories:
            snaps = traj.snapshots
            for s in range(len(snaps) - 1):
                u_t = snaps[s]
                u_tp1 = snaps[s + 1]

                t_in = torch.from_numpy(u_t).unsqueeze(0).unsqueeze(0).float()
                u_p = model(t_in).squeeze().numpy().astype(np.float64)

                all_u_in.append(u_t)
                all_u_ref.append(u_tp1)
                all_u_pred.append(u_p)
                all_types.append(traj.generator_type)

    # 1. Model Errors
    diffs_model = [p - r for p, r in zip(all_u_pred, all_u_ref, strict=False)]
    mse_model = float(np.mean([np.mean(d**2) for d in diffs_model]))
    rmse_model = float(np.sqrt(mse_model))
    rel_l2_sample_model = float(
        np.mean(
            [
                compute_l2_error(p, r, relative=True)
                for p, r in zip(all_u_pred, all_u_ref, strict=False)
            ]
        )
    )
    sum_sq_diff_m = sum(np.sum(d**2) for d in diffs_model)
    sum_sq_ref = sum(np.sum(r**2) for r in all_u_ref)
    rel_l2_frob_model = float(np.sqrt(sum_sq_diff_m / max(1e-12, sum_sq_ref)))
    max_err_model = float(
        np.max(
            [
                compute_max_error(p, r)
                for p, r in zip(all_u_pred, all_u_ref, strict=False)
            ]
        )
    )

    # 2. Identity Baseline Errors: predict u(t+0.005) = u(t)
    diffs_id = [u_in - r for u_in, r in zip(all_u_in, all_u_ref, strict=False)]
    mse_id = float(np.mean([np.mean(d**2) for d in diffs_id]))
    rmse_id = float(np.sqrt(mse_id))
    rel_l2_sample_id = float(
        np.mean(
            [
                compute_l2_error(u_in, r, relative=True)
                for u_in, r in zip(all_u_in, all_u_ref, strict=False)
            ]
        )
    )
    sum_sq_diff_id = sum(np.sum(d**2) for d in diffs_id)
    rel_l2_frob_id = float(np.sqrt(sum_sq_diff_id / max(1e-12, sum_sq_ref)))
    max_err_id = float(
        np.max(
            [
                compute_max_error(u_in, r)
                for u_in, r in zip(all_u_in, all_u_ref, strict=False)
            ]
        )
    )

    # Subset metrics (Fourier vs Gaussian)
    f_idxs = [i for i, t in enumerate(all_types) if t == "fourier"]
    g_idxs = [i for i, t in enumerate(all_types) if t == "gaussian"]

    def _calc_subset(idxs: list[int]) -> dict[str, float]:
        sub_d = [diffs_model[i] for i in idxs]
        sub_p = [all_u_pred[i] for i in idxs]
        sub_r = [all_u_ref[i] for i in idxs]
        sub_mse = float(np.mean([np.mean(d**2) for d in sub_d]))
        sub_sq_d = sum(np.sum(d**2) for d in sub_d)
        sub_sq_r = sum(np.sum(r**2) for r in sub_r)
        return {
            "mse": sub_mse,
            "rmse": float(np.sqrt(sub_mse)),
            "rel_l2_sample": float(
                np.mean(
                    [
                        compute_l2_error(p, r, relative=True)
                        for p, r in zip(sub_p, sub_r, strict=False)
                    ]
                )
            ),
            "rel_l2_frob": float(np.sqrt(sub_sq_d / max(1e-12, sub_sq_r))),
            "max_error": float(
                np.max(
                    [
                        compute_max_error(p, r)
                        for p, r in zip(sub_p, sub_r, strict=False)
                    ]
                )
            ),
        }

    fourier_metrics = _calc_subset(f_idxs)
    gaussian_metrics = _calc_subset(g_idxs)

    # Boundary enforcement check
    boundary_violations = []
    for p in all_u_pred:
        b_max = max(
            np.max(np.abs(p[0, :])),
            np.max(np.abs(p[-1, :])),
            np.max(np.abs(p[:, 0])),
            np.max(np.abs(p[:, -1])),
        )
        boundary_violations.append(float(b_max))
    max_boundary_violation = float(np.max(boundary_violations))

    # Temperature extremes
    min_temp_all = float(np.min([np.min(p) for p in all_u_pred]))
    max_temp_all = float(np.max([np.max(p) for p in all_u_pred]))
    gauss_preds = [all_u_pred[i] for i in g_idxs]
    min_temp_gauss = float(np.min([np.min(p) for p in gauss_preds]))
    max_temp_gauss = float(np.max([np.max(p) for p in gauss_preds]))

    # Gaussian interior negative cell count
    total_g_interior = len(gauss_preds) * 49 * 49
    total_g_nodes = len(gauss_preds) * 51 * 51
    g_neg_tol = sum(int(np.sum(p[1:-1, 1:-1] < -1e-5)) for p in gauss_preds)
    g_neg_strict = sum(int(np.sum(p[1:-1, 1:-1] < 0.0)) for p in gauss_preds)

    # Spatial Laplacian metrics
    lap_diffs: list[np.ndarray] = []
    lap_refs: list[np.ndarray] = []
    for p, r in zip(all_u_pred, all_u_ref, strict=False):
        tp = torch.from_numpy(p).unsqueeze(0).unsqueeze(0).float()
        tr = torch.from_numpy(r).unsqueeze(0).unsqueeze(0).float()
        lp = compute_discrete_laplacian_torch(tp, dx, dy).squeeze().numpy()
        lr = compute_discrete_laplacian_torch(tr, dx, dy).squeeze().numpy()
        lap_diffs.append(lp - lr)
        lap_refs.append(lr)

    lap_mse = float(np.mean([np.mean(ld**2) for ld in lap_diffs]))
    lap_rmse = float(np.sqrt(lap_mse))
    lap_rel_l2_sample = float(
        np.mean(
            [
                np.linalg.norm(ld) / max(1e-12, np.linalg.norm(lr))
                for ld, lr in zip(lap_diffs, lap_refs, strict=False)
            ]
        )
    )
    lap_rel_l2_frob = float(
        np.sqrt(
            sum(np.sum(ld**2) for ld in lap_diffs)
            / max(1e-12, sum(np.sum(lr**2) for lr in lap_refs))
        )
    )

    # Energy dissipation delta H = H(t+dt) - H(t)
    e_in = [float(np.sum(u**2) * dx * dy) for u in all_u_in]
    e_ref = [float(np.sum(r**2) * dx * dy) for r in all_u_ref]
    e_pred = [float(np.sum(p**2) * dx * dy) for p in all_u_pred]
    dh_ref = np.array(e_ref) - np.array(e_in)
    dh_pred = np.array(e_pred) - np.array(e_in)
    dh_mae = float(np.mean(np.abs(dh_pred - dh_ref)))

    return {
        "model": {
            "mse": mse_model,
            "rmse": rmse_model,
            "rel_l2_sample": rel_l2_sample_model,
            "rel_l2_frob": rel_l2_frob_model,
            "max_error": max_err_model,
            "fourier": fourier_metrics,
            "gaussian": gaussian_metrics,
            "boundary_violation_max": max_boundary_violation,
            "min_temp_all": min_temp_all,
            "max_temp_all": max_temp_all,
            "min_temp_gaussian": min_temp_gauss,
            "max_temp_gaussian": max_temp_gauss,
            "gaussian_neg_cells_strict": g_neg_strict,
            "gaussian_neg_cells_tol": g_neg_tol,
            "gaussian_neg_ratio_interior": g_neg_tol / total_g_interior,
            "gaussian_neg_ratio_total": g_neg_tol / total_g_nodes,
            "laplacian_mse": lap_mse,
            "laplacian_rmse": lap_rmse,
            "laplacian_rel_l2_sample": lap_rel_l2_sample,
            "laplacian_rel_l2_frob": lap_rel_l2_frob,
            "energy_dissipation_mae": dh_mae,
        },
        "identity_baseline": {
            "mse": mse_id,
            "rmse": rmse_id,
            "rel_l2_sample": rel_l2_sample_id,
            "rel_l2_frob": rel_l2_frob_id,
            "max_error": max_err_id,
        },
    }


def evaluate_autoregressive_rollouts(
    model: torch.nn.Module,
    test_trajectories: list[Any],
    dx: float,
    dy: float,
) -> dict[str, Any]:
    """Execute multi-step autoregressive rollouts up to 100 steps (0.5s).

    Evaluates specific milestone horizons:
    - 20 steps (t = 0.1s)
    - 40 steps (t = 0.2s)
    - 100 steps (t = 0.5s)
    """
    evaluator = RolloutEvaluator(model, device="cpu")

    results_per_traj = []
    for traj in test_trajectories:
        res = evaluator.evaluate_trajectory(traj)
        results_per_traj.append(res)

    # Aggregate across trajectories
    all_rel_curves = np.array(
        [r["relative_l2_errors"] for r in results_per_traj]
    )  # (N, 101)
    mean_rel_curve = np.mean(all_rel_curves, axis=0).tolist()

    f_trajs = [r for r in results_per_traj if r["generator_type"] == "fourier"]
    g_trajs = [r for r in results_per_traj if r["generator_type"] == "gaussian"]

    f_rel_curve = np.mean([r["relative_l2_errors"] for r in f_trajs], axis=0).tolist()
    g_rel_curve = np.mean([r["relative_l2_errors"] for r in g_trajs], axis=0).tolist()

    # Specific milestone step comparisons: 20, 40, 100
    milestone_steps = [20, 40, 100]
    milestone_times = [0.1, 0.2, 0.5]
    milestone_data = {}

    for step, phys_t in zip(milestone_steps, milestone_times, strict=False):
        step_l2_all = float(mean_rel_curve[step])
        step_l2_f = float(f_rel_curve[step])
        step_l2_g = float(g_rel_curve[step])

        # Peak and energy comparison at step
        g_peaks_ref = [float(np.max(r["ground_truth"][step])) for r in g_trajs]
        g_peaks_pred = [float(np.max(r["predicted"][step])) for r in g_trajs]
        g_energies_ref = [
            float(np.sum(r["ground_truth"][step] ** 2) * dx * dy) for r in g_trajs
        ]
        g_energies_pred = [
            float(np.sum(r["predicted"][step] ** 2) * dx * dy) for r in g_trajs
        ]

        # Gaussian negative cells at this rollout step
        neg_cells_step = sum(
            int(np.sum(r["predicted"][step][1:-1, 1:-1] < -1e-5)) for r in g_trajs
        )
        min_temp_step = float(min(np.min(r["predicted"][step]) for r in g_trajs))

        milestone_data[f"step_{step}"] = {
            "physical_time_s": phys_t,
            "relative_l2_overall": step_l2_all,
            "relative_l2_fourier": step_l2_f,
            "relative_l2_gaussian": step_l2_g,
            "gaussian_peak_ref": float(np.mean(g_peaks_ref)),
            "gaussian_peak_pred": float(np.mean(g_peaks_pred)),
            "gaussian_energy_ref": float(np.mean(g_energies_ref)),
            "gaussian_energy_pred": float(np.mean(g_energies_pred)),
            "gaussian_neg_cells": neg_cells_step,
            "gaussian_min_temp": min_temp_step,
        }

    # Full 100-step peak and energy trajectories
    g_peak_traj_ref = np.mean(
        [np.max(r["ground_truth"], axis=(1, 2)) for r in g_trajs], axis=0
    ).tolist()
    g_peak_traj_pred = np.mean(
        [np.max(r["predicted"], axis=(1, 2)) for r in g_trajs], axis=0
    ).tolist()
    g_mean_traj_ref = np.mean(
        [np.mean(r["ground_truth"], axis=(1, 2)) for r in g_trajs], axis=0
    ).tolist()
    g_mean_traj_pred = np.mean(
        [np.mean(r["predicted"], axis=(1, 2)) for r in g_trajs], axis=0
    ).tolist()

    return {
        "mean_relative_l2_curve": mean_rel_curve,
        "fourier_relative_l2_curve": f_rel_curve,
        "gaussian_relative_l2_curve": g_rel_curve,
        "milestones": milestone_data,
        "gaussian_peak_ref": g_peak_traj_ref,
        "gaussian_peak_pred": g_peak_traj_pred,
        "gaussian_mean_ref": g_mean_traj_ref,
        "gaussian_mean_pred": g_mean_traj_pred,
    }


def main() -> None:
    args = parse_args()
    print("=" * 80)
    print("MILESTONE 5: ONE-STEP NEURAL SURROGATE TRAINING & EVALUATION")
    print("=" * 80)

    repo_root = Path(__file__).resolve().parent.parent
    dataset_dir = (
        (repo_root / args.dataset_dir)
        if not Path(args.dataset_dir).is_absolute()
        else Path(args.dataset_dir)
    )
    models_dir = (
        (repo_root / args.models_dir)
        if not Path(args.models_dir).is_absolute()
        else Path(args.models_dir)
    )
    figures_dir = (
        (repo_root / args.figures_dir)
        if not Path(args.figures_dir).is_absolute()
        else Path(args.figures_dir)
    )

    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    # 1. Load dataset splits
    print(f"Loading one-step dataset from {dataset_dir}...")
    splits = load_dataset(dataset_dir)
    config = splits.train[0].config
    dx, dy = config.dx, config.dy

    train_ds = HeatDiffusionDataset(splits.train, mode="step")
    val_ds = HeatDiffusionDataset(splits.val, mode="step")
    test_trajectories = splits.test

    print(
        f"Loaded: {len(train_ds)} train pairs, {len(val_ds)} val pairs, "
        f"{len(test_trajectories)} test simulations."
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # 2. Initialize OneStepHeatResNet2D Model
    model = OneStepHeatResNet2D(
        in_channels=1,
        hidden_channels=32,
        num_blocks=3,
        enforce_zero_boundary=True,
    )
    print(
        f"Initialized OneStepHeatResNet2D with {model.count_parameters():,} "
        "trainable parameters."
    )

    # 3. Train Surrogate Model with MSE Loss only
    trainer = SurrogateTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.lr,
        weight_decay=1e-4,
        lambda_laplacian=0.0,  # Pure MSE objective
        dx=dx,
        dy=dy,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer,
        T_max=args.epochs,
        eta_min=1e-5,
    )

    checkpoint_path = models_dir / "surrogate_onestep_ftcs.pt"
    print(
        f"Beginning training for {args.epochs} epochs "
        f"(checkpoint dest: {checkpoint_path.name})..."
    )
    history = trainer.fit(
        epochs=args.epochs,
        checkpoint_path=checkpoint_path,
        scheduler=scheduler,
    )

    # Load best validation checkpoint
    trainer.load_checkpoint(checkpoint_path)
    print(
        f"Training complete. Best checkpoint at epoch {history.best_epoch} "
        f"with validation loss: {history.best_val_loss:.6e}."
    )

    # 4. One-Step Test Evaluation
    print("\nExecuting comprehensive one-step held-out test evaluation...")
    eval_one_step_results = evaluate_one_step(
        model=model,
        test_trajectories=test_trajectories,
        dx=dx,
        dy=dy,
    )

    m_res = eval_one_step_results["model"]
    id_res = eval_one_step_results["identity_baseline"]

    print("\n" + "=" * 90)
    print("ONE-STEP PREDICTION TEST RESULTS (dt = 0.005s)")
    print("=" * 90)
    print(f"{'Metric':<35} | {'Neural Surrogate':<22} | {'Identity Baseline':<22}")
    print("-" * 90)
    print(f"{'Field MSE':<35} | {m_res['mse']:<22.6e} | {id_res['mse']:<22.6e}")
    print(f"{'Field RMSE':<35} | {m_res['rmse']:<22.6f} | {id_res['rmse']:<22.6f}")
    print(
        f"{'Relative L2 (Sample Mean)':<35} | "
        f"{m_res['rel_l2_sample'] * 100:<21.4f}% | "
        f"{id_res['rel_l2_sample'] * 100:<21.4f}%"
    )
    print(
        f"{'Relative L2 (Frobenius)':<35} | "
        f"{m_res['rel_l2_frob'] * 100:<21.4f}% | "
        f"{id_res['rel_l2_frob'] * 100:<21.4f}%"
    )
    print(
        f"{'Maximum Absolute Error':<35} | {m_res['max_error']:<22.6f} | "
        f"{id_res['max_error']:<22.6f}"
    )
    print(f"{'Laplacian RMSE':<35} | {m_res['laplacian_rmse']:<22.6f} | {'N/A':<22}")
    print(
        f"{'Laplacian Relative L2 [%]':<35} | "
        f"{m_res['laplacian_rel_l2_sample'] * 100:<21.4f}% | {'N/A':<22}"
    )
    print(
        f"{'Boundary Violation (max)':<35} | "
        f"{m_res['boundary_violation_max']:<22.6e} | {'0.000000':<22}"
    )
    print(
        f"{'Energy Dissipation MAE (dH)':<35} | "
        f"{m_res['energy_dissipation_mae']:<22.6e} | {'N/A':<22}"
    )
    print("-" * 90)
    f_l2 = m_res["fourier"]["rel_l2_sample"] * 100
    g_l2 = m_res["gaussian"]["rel_l2_sample"] * 100
    print(f"Fourier Subset MSE   : {m_res['fourier']['mse']:.6e} (Rel L2: {f_l2:.3f}%)")
    print(
        f"Gaussian Subset MSE  : {m_res['gaussian']['mse']:.6e} (Rel L2: {g_l2:.3f}%)"
    )
    print(
        f"Gaussian Min Temp    : {m_res['min_temp_gaussian']:.6f} "
        f"({m_res['gaussian_neg_cells_tol']} cells < -1e-5)"
    )

    # 5. Multi-Step Autoregressive Rollout Evaluation
    print("\nExecuting multi-step autoregressive rollouts (100 steps, dt=0.005s)...")
    rollout_results = evaluate_autoregressive_rollouts(
        model=model,
        test_trajectories=test_trajectories,
        dx=dx,
        dy=dy,
    )

    print("\n" + "=" * 90)
    print("AUTOREGRESSIVE ROLLOUT MILESTONE HORIZONS")
    print("=" * 90)
    hdr_ro = (
        f"{'Horizon':<15} | {'Steps':<6} | {'Rel L2 (All)':<14} | "
        f"{'Fourier L2':<12} | {'Gaussian L2':<12} | {'Gauss Peak Err':<16}"
    )
    print(hdr_ro)
    print("-" * 90)
    for k_step in [20, 40, 100]:
        m = rollout_results["milestones"][f"step_{k_step}"]
        t_sec = m["physical_time_s"]
        peak_err = m["gaussian_peak_pred"] - m["gaussian_peak_ref"]
        print(
            f"t = {t_sec:<4.2f}s        | {k_step:<6} | "
            f"{m['relative_l2_overall'] * 100:<13.3f}% | "
            f"{m['relative_l2_fourier'] * 100:<11.3f}% | "
            f"{m['relative_l2_gaussian'] * 100:<11.3f}% | "
            f"{peak_err:<+16.4f}"
        )

    # 6. Visualizations Generation
    epochs_range = range(1, args.epochs + 1)

    # Figure 1: Training History
    _fig1, ax1 = plt.subplots(1, 2, figsize=(12, 5))
    ax1[0].plot(
        epochs_range,
        history.train_losses,
        label="Train Loss (MSE)",
        color="#1f77b4",
        lw=2,
    )
    ax1[0].plot(
        epochs_range, history.val_losses, label="Val Loss (MSE)", color="#d62728", lw=2
    )
    ax1[0].axvline(
        history.best_epoch,
        color="black",
        ls="--",
        label=f"Best Checkpoint (Ep {history.best_epoch})",
    )
    ax1[0].set_yscale("log")
    ax1[0].set_title(
        "Milestone 5 One-Step Surrogate Training Loss", fontsize=11, fontweight="bold"
    )
    ax1[0].set_xlabel("Epoch", fontsize=10)
    ax1[0].set_ylabel("MSE Loss", fontsize=10)
    ax1[0].grid(True, which="both", ls="--", alpha=0.5)
    ax1[0].legend(fontsize=9)

    # Learning rate schedule
    lrs = [args.lr * 0.5 * (1 + np.cos(np.pi * e / args.epochs)) for e in epochs_range]
    ax1[1].plot(epochs_range, lrs, color="#2ca02c", lw=2)
    ax1[1].set_title(
        "Cosine Annealing Learning Rate Schedule", fontsize=11, fontweight="bold"
    )
    ax1[1].set_xlabel("Epoch", fontsize=10)
    ax1[1].set_ylabel("Learning Rate", fontsize=10)
    ax1[1].grid(True, ls="--", alpha=0.5)

    plt.tight_layout()
    fig1_path = figures_dir / "onestep_training_curves.png"
    plt.savefig(fig1_path, dpi=200)
    plt.close()
    print(f"Saved: {fig1_path}")

    # Figure 2: Rollout Error & Peak Temperature Progression (100 steps)
    steps_arr = np.arange(101)
    phys_times = steps_arr * config.time_step

    _fig2, ax2 = plt.subplots(1, 2, figsize=(14, 5))
    ax2[0].plot(
        phys_times,
        np.array(rollout_results["mean_relative_l2_curve"]) * 100,
        label="Overall Test Ensemble",
        color="#1f77b4",
        lw=2.5,
    )
    ax2[0].plot(
        phys_times,
        np.array(rollout_results["fourier_relative_l2_curve"]) * 100,
        label="Fourier Mode Subset",
        color="#2ca02c",
        lw=2,
        ls="--",
    )
    ax2[0].plot(
        phys_times,
        np.array(rollout_results["gaussian_relative_l2_curve"]) * 100,
        label="Gaussian Source Subset",
        color="#ff7f0e",
        lw=2,
        ls=":",
    )

    # Milestone markers
    for step in [20, 40, 100]:
        t_m = step * config.time_step
        err_m = rollout_results["mean_relative_l2_curve"][step] * 100
        ax2[0].scatter([t_m], [err_m], color="red", s=50, zorder=5)
        ax2[0].annotate(
            f"{err_m:.2f}%\n({step} steps)",
            (t_m, err_m),
            textcoords="offset points",
            xytext=(-15, 12),
            fontsize=8,
            fontweight="bold",
        )

    ax2[0].set_title(
        "100-Step Autoregressive Rollout Relative $L_2$ Error [%]",
        fontsize=11,
        fontweight="bold",
    )
    ax2[0].set_xlabel("Physical Time t [s]", fontsize=10)
    ax2[0].set_ylabel("Relative $L_2$ Error [%]", fontsize=10)
    ax2[0].grid(True, ls="--", alpha=0.5)
    ax2[0].legend(fontsize=9)

    # Gaussian Peak Temperature Evolution
    ax2[1].plot(
        phys_times,
        rollout_results["gaussian_peak_ref"],
        label="FTCS Reference $u_{\\rm max}(t)$",
        color="black",
        lw=2.5,
    )
    ax2[1].plot(
        phys_times,
        rollout_results["gaussian_peak_pred"],
        label="One-Step Surrogate Rollout",
        color="#d62728",
        lw=2,
        ls="--",
    )
    ax2[1].set_title(
        "Gaussian Peak Temperature Decay $u_{\\rm max}(t)$",
        fontsize=11,
        fontweight="bold",
    )
    ax2[1].set_xlabel("Physical Time t [s]", fontsize=10)
    ax2[1].set_ylabel("Peak Temperature [K]", fontsize=10)
    ax2[1].grid(True, ls="--", alpha=0.5)
    ax2[1].legend(fontsize=9)

    plt.tight_layout()
    fig2_path = figures_dir / "onestep_rollout_evolution.png"
    plt.savefig(fig2_path, dpi=200)
    plt.close()
    print(f"Saved: {fig2_path}")

    # Figure 3: Spatial Predictions and Error Maps at Milestone Horizons
    target_sim = next(s for s in test_trajectories if s.generator_type == "gaussian")
    rollout_evaluator = RolloutEvaluator(model, device="cpu")
    pred_states = rollout_evaluator.predict_rollout(
        target_sim.initial_state, num_steps=100
    )
    gt_states = target_sim.snapshots

    _fig3, axes3 = plt.subplots(3, 4, figsize=(16, 11))
    step_list = [0, 20, 40, 100]
    col_titles = [
        "t = 0.0s (Initial u0)",
        "t = 0.1s (Step 20)",
        "t = 0.2s (Step 40)",
        "t = 0.5s (Step 100)",
    ]

    for col, (step_idx, col_title) in enumerate(
        zip(step_list, col_titles, strict=False)
    ):
        # Row 0: Reference
        im_ref = axes3[0, col].imshow(
            gt_states[step_idx], cmap="inferno", origin="lower", vmin=0.0, vmax=0.7
        )
        axes3[0, col].set_title(
            f"FTCS Reference\n{col_title}", fontsize=10, fontweight="bold"
        )
        plt.colorbar(im_ref, ax=axes3[0, col], fraction=0.046)

        # Row 1: Predicted
        im_pred = axes3[1, col].imshow(
            pred_states[step_idx], cmap="inferno", origin="lower", vmin=0.0, vmax=0.7
        )
        axes3[1, col].set_title(
            f"Surrogate Rollout\n{col_title}", fontsize=10, fontweight="bold"
        )
        plt.colorbar(im_pred, ax=axes3[1, col], fraction=0.046)

        # Row 2: Signed Error
        err_map = pred_states[step_idx] - gt_states[step_idx]
        v_limit = max(0.01, float(np.max(np.abs(err_map))))
        im_err = axes3[2, col].imshow(
            err_map, cmap="coolwarm", origin="lower", vmin=-v_limit, vmax=v_limit
        )
        max_e = float(np.max(np.abs(err_map)))
        axes3[2, col].set_title(
            f"Signed Error (Pred - Ref)\nMax |Err|: {max_e:.4f}",
            fontsize=10,
        )
        plt.colorbar(im_err, ax=axes3[2, col], fraction=0.046)

    plt.tight_layout()
    fig3_path = figures_dir / "onestep_field_comparison.png"
    plt.savefig(fig3_path, dpi=200)
    plt.close()
    print(f"Saved: {fig3_path}")

    # 7. Save Comprehensive Milestone 5 Report JSON
    report_payload = {
        "metadata": {
            "milestone": "5",
            "model_architecture": "OneStepHeatResNet2D",
            "parameters": model.count_parameters(),
            "time_step_dt": config.time_step,
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "best_epoch": history.best_epoch,
            "best_val_loss": history.best_val_loss,
            "checkpoint_path": str(checkpoint_path),
        },
        "training_history": history.to_dict(),
        "one_step_evaluation": eval_one_step_results,
        "rollout_evaluation": rollout_results,
    }

    report_file = models_dir / "onestep_surrogate_evaluation_report.json"
    with open(report_file, "w") as f:
        json.dump(report_payload, f, indent=2)
    print(f"\nSaved complete Milestone 5 evaluation report: {report_file}")
    print("Milestone 5 Training and Evaluation Completed Successfully.")


if __name__ == "__main__":
    main()
