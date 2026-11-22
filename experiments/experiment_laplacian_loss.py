"""Milestone 4.2: Physics-Informed Spatial Laplacian Loss Regularization.

Objective:
Investigate whether penalizing spatial Laplacian discrepancies:
    L_total = L_MSE + lambda * L_Laplacian
improves the physical derivative fidelity and multi-step stability
of HeatResNet2D without altering the CNN architecture.

Experimental Configurations:
1. Baseline (reproduced): lambda = 0.0
2. Physics-Informed: lambda = 0.001
3. Physics-Informed: lambda = 0.01
Also benchmarks against original baseline checkpoint.
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
from physics_ml.models.surrogate import HeatResNet2D
from physics_ml.models.trainer import (
    SurrogateTrainer,
    TrainingHistory,
    compute_discrete_laplacian_torch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Milestone 4.2: Physics-Informed Loss Experiment"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="outputs/datasets/heat_diffusion_v1",
        help="Path to dataset directory.",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="outputs/models",
        help="Directory to save/load model checkpoints.",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default="outputs/figures",
        help="Directory to save diagnostic visualization figures.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
        help="Training epochs per configuration (default: 60).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (default: 16).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible initializations.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seed across PyTorch and NumPy for exact reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model_configuration(
    lambda_lap: float,
    train_loader: DataLoader[dict[str, torch.Tensor | str]],
    val_loader: DataLoader[dict[str, torch.Tensor | str]],
    dx: float,
    dy: float,
    epochs: int,
    checkpoint_path: Path,
    seed: int,
) -> tuple[HeatResNet2D, TrainingHistory]:
    """Train HeatResNet2D with specified lambda_laplacian and CosineAnnealing."""
    set_seed(seed)

    model = HeatResNet2D(in_channels=1, hidden_channels=32, num_blocks=3)
    trainer = SurrogateTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=1e-3,
        weight_decay=1e-4,
        lambda_laplacian=lambda_lap,
        dx=dx,
        dy=dy,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer,
        T_max=epochs,
        eta_min=1e-5,
    )

    history = trainer.fit(
        epochs=epochs,
        checkpoint_path=checkpoint_path,
        scheduler=scheduler,
    )

    # Load best checkpoint based on validation loss
    trainer.load_checkpoint(checkpoint_path)
    return model, history


def evaluate_model_full(
    model: HeatResNet2D,
    test_trajectories: list[Any],
    dx: float,
    dy: float,
    nx: int,
    ny: int,
) -> dict[str, Any]:
    """Comprehensive test evaluation across all audited physical metrics."""
    model.eval()

    all_u_in: list[np.ndarray] = []
    all_u_ref: list[np.ndarray] = []
    all_u_pred: list[np.ndarray] = []
    all_types: list[str] = []

    for traj in test_trajectories:
        snaps = traj.snapshots
        n_pairs = len(snaps) - 1
        for s in range(n_pairs):
            u_t = snaps[s]
            u_tp1 = snaps[s + 1]

            t_in = torch.from_numpy(u_t).unsqueeze(0).unsqueeze(0).float()
            with torch.no_grad():
                u_p = model(t_in).squeeze().numpy().astype(np.float64)

            all_u_in.append(u_t)
            all_u_ref.append(u_tp1)
            all_u_pred.append(u_p)
            all_types.append(traj.generator_type)

    # 1. Temperature field metrics
    diffs = [p - r for p, r in zip(all_u_pred, all_u_ref, strict=False)]
    mse = float(np.mean([np.mean(d**2) for d in diffs]))
    rmse = float(np.sqrt(mse))
    rel_l2 = float(
        np.mean(
            [
                compute_l2_error(p, r, relative=True)
                for p, r in zip(all_u_pred, all_u_ref, strict=False)
            ]
        )
    )
    max_err = float(
        np.max(
            [
                compute_max_error(p, r)
                for p, r in zip(all_u_pred, all_u_ref, strict=False)
            ]
        )
    )

    # Subset metrics
    f_idxs = [i for i, t in enumerate(all_types) if t == "fourier"]
    g_idxs = [i for i, t in enumerate(all_types) if t == "gaussian"]

    def _sub_stats(idxs: list[int]) -> dict[str, float]:
        sub_d = [diffs[i] for i in idxs]
        sub_p = [all_u_pred[i] for i in idxs]
        sub_r = [all_u_ref[i] for i in idxs]
        sub_mse = float(np.mean([np.mean(d**2) for d in sub_d]))
        return {
            "mse": sub_mse,
            "rmse": float(np.sqrt(sub_mse)),
            "relative_l2": float(
                np.mean(
                    [
                        compute_l2_error(p, r, relative=True)
                        for p, r in zip(sub_p, sub_r, strict=False)
                    ]
                )
            ),
            "max_error": float(
                np.max(
                    [
                        compute_max_error(p, r)
                        for p, r in zip(sub_p, sub_r, strict=False)
                    ]
                )
            ),
        }

    fourier_stats = _sub_stats(f_idxs)
    gaussian_stats = _sub_stats(g_idxs)

    # 2. Discrete Spatial Laplacian Metrics
    lap_diffs: list[np.ndarray] = []
    lap_refs: list[np.ndarray] = []
    for p, r in zip(all_u_pred, all_u_ref, strict=False):
        t_p = torch.from_numpy(p).unsqueeze(0).unsqueeze(0).float()
        t_r = torch.from_numpy(r).unsqueeze(0).unsqueeze(0).float()
        lp = compute_discrete_laplacian_torch(t_p, dx, dy).squeeze().numpy()
        lr = compute_discrete_laplacian_torch(t_r, dx, dy).squeeze().numpy()
        lap_diffs.append(lp - lr)
        lap_refs.append(lr)

    lap_mse = float(np.mean([np.mean(ld**2) for ld in lap_diffs]))
    lap_rmse = float(np.sqrt(lap_mse))
    lap_rel_l2 = float(
        np.mean(
            [
                np.linalg.norm(ld) / max(1e-12, np.linalg.norm(lr))
                for ld, lr in zip(lap_diffs, lap_refs, strict=False)
            ]
        )
    )

    # 3. Boundary distance profile
    dist_map = np.zeros((nx, ny), dtype=int)
    for i in range(nx):
        for j in range(ny):
            dist_map[i, j] = min(i, j, nx - 1 - i, ny - 1 - j)
    max_d = int(np.max(dist_map))

    mae_by_d = np.zeros(max_d + 1)
    for d in range(max_d + 1):
        m = dist_map == d
        d_errs = np.concatenate([np.abs(d_i[m]) for d_i in diffs])
        mae_by_d[d] = float(np.mean(d_errs))

    # 4. Gaussian Negative Prediction Count & Min Temp
    gauss_preds = [all_u_pred[i] for i in g_idxs]
    min_gauss_val = float(np.min([np.min(p) for p in gauss_preds]))
    neg_cells = sum(int(np.sum(p < -1e-5)) for p in gauss_preds)
    total_g_cells = len(gauss_preds) * nx * ny

    # 5. Energy Dissipation Fidelity (MAE of delta H)
    energy_in = [float(np.sum(u**2) * dx * dy) for u in all_u_in]
    energy_ref = [float(np.sum(u**2) * dx * dy) for u in all_u_ref]
    energy_pred = [float(np.sum(u**2) * dx * dy) for u in all_u_pred]

    delta_h_ref = np.array(energy_ref) - np.array(energy_in)
    delta_h_pred = np.array(energy_pred) - np.array(energy_in)
    energy_diss_mae = float(np.mean(np.abs(delta_h_pred - delta_h_ref)))

    # 6. Ten-Step Autoregressive Rollout Evaluation
    rollout_evaluator = RolloutEvaluator(model, device="cpu")
    ensemble_rollout = rollout_evaluator.evaluate_ensemble(test_trajectories)
    rollout_rel_l2 = ensemble_rollout["mean_relative_l2"]
    final_rollout_l2 = float(rollout_rel_l2[-1])

    return {
        "mse": mse,
        "rmse": rmse,
        "relative_l2": rel_l2,
        "max_error": max_err,
        "fourier": fourier_stats,
        "gaussian": gaussian_stats,
        "laplacian_mse": lap_mse,
        "laplacian_rmse": lap_rmse,
        "laplacian_relative_l2": lap_rel_l2,
        "mae_by_distance": mae_by_d.tolist(),
        "d1_mae": float(mae_by_d[1]),
        "d2_mae": float(mae_by_d[2]),
        "d15_mae": float(mae_by_d[15]),
        "gaussian_min_temp": min_gauss_val,
        "gaussian_neg_cells": neg_cells,
        "gaussian_neg_ratio": neg_cells / total_g_cells,
        "energy_dissipation_mae": energy_diss_mae,
        "rollout_rel_l2_curve": rollout_rel_l2,
        "final_rollout_rel_l2": final_rollout_l2,
        "rollout_amplification": float(final_rollout_l2 / max(1e-12, rel_l2)),
    }


def main() -> None:
    args = parse_args()
    print("=" * 80)
    print("MILESTONE 4.2: PHYSICS-INFORMED SPATIAL LAPLACIAN LOSS EXPERIMENT")
    print("=" * 80)

    # 1. Load dataset splits
    dataset_dir = Path(args.dataset_dir)
    splits = load_dataset(dataset_dir)
    config = splits.train[0].config
    dx, dy = config.dx, config.dy
    nx, ny = config.grid_resolution_x, config.grid_resolution_y

    train_ds = HeatDiffusionDataset(splits.train, mode="step")
    val_ds = HeatDiffusionDataset(splits.val, mode="step")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 2. Experimental Configurations
    experiments = [
        {
            "name": "Baseline (Reproduced)",
            "lambda": 0.0,
            "ckpt": models_dir / "surrogate_baseline_reproduced.pt",
        },
        {
            "name": "Physics-Informed (lambda=0.001)",
            "lambda": 0.001,
            "ckpt": models_dir / "surrogate_laplacian_lambda_0.001.pt",
        },
        {
            "name": "Physics-Informed (lambda=0.01)",
            "lambda": 0.01,
            "ckpt": models_dir / "surrogate_laplacian_lambda_0.01.pt",
        },
    ]

    trained_models: dict[str, HeatResNet2D] = {}
    histories: dict[str, TrainingHistory] = {}

    for exp in experiments:
        name = exp["name"]
        lam = exp["lambda"]
        ckpt_path = exp["ckpt"]

        print(f"\n>>> Training Configuration: {name} (lambda = {lam})")
        print(f"    Target checkpoint: {ckpt_path.name}")
        model, hist = train_model_configuration(
            lambda_lap=lam,
            train_loader=train_loader,
            val_loader=val_loader,
            dx=dx,
            dy=dy,
            epochs=args.epochs,
            checkpoint_path=ckpt_path,
            seed=args.seed,
        )
        trained_models[name] = model
        histories[name] = hist
        print(
            f"    Complete: Best Val Loss = {hist.best_val_loss:.6e} "
            f"at epoch {hist.best_epoch}"
        )

    # Also load the original baseline checkpoint for comparison
    orig_ckpt_path = models_dir / "best_surrogate.pt"
    orig_model = HeatResNet2D(in_channels=1, hidden_channels=32, num_blocks=3)
    orig_checkpoint = torch.load(orig_ckpt_path, map_location="cpu", weights_only=False)
    orig_model.load_state_dict(orig_checkpoint["model_state_dict"])
    trained_models["Original Baseline (M4)"] = orig_model

    # 3. Comprehensive Evaluation of all configurations on test split
    print("\n" + "=" * 80)
    print("EVALUATING ALL CONFIGURATIONS ON HELD-OUT TEST DATA")
    print("=" * 80)

    eval_results: dict[str, dict[str, Any]] = {}
    for name, model in trained_models.items():
        print(f"Evaluating {name}...")
        eval_results[name] = evaluate_model_full(
            model=model,
            test_trajectories=splits.test,
            dx=dx,
            dy=dy,
            nx=nx,
            ny=ny,
        )

    # 4. Generate Comparison Table
    print("\n" + "=" * 80)
    print("MILESTONE 4.2 COMPREHENSIVE EXPERIMENT COMPARISON TABLE")
    print("=" * 80)
    hdr = (
        f"{'Model Configuration':<30} | {'Test MSE':<12} | {'Rel L2 [%]':<11} | "
        f"{'Lap RMSE':<10} | {'Lap Rel L2 [%]':<14} | {'10-Step Rollout':<15}"
    )
    print(hdr)
    print("-" * len(hdr))
    for name in [
        "Original Baseline (M4)",
        "Baseline (Reproduced)",
        "Physics-Informed (lambda=0.001)",
        "Physics-Informed (lambda=0.01)",
    ]:
        res = eval_results[name]
        l_rel = res["laplacian_relative_l2"] * 100
        ro_rel = res["final_rollout_rel_l2"] * 100
        print(
            f"{name:<30} | {res['mse']:<12.6e} | {res['relative_l2'] * 100:<11.3f} | "
            f"{res['laplacian_rmse']:<10.4f} | {l_rel:<14.3f} | {ro_rel:<15.3f}"
        )

    # 5. Diagnostic Figures Generation
    # Figure 1: Training & Validation Loss Curves across Epochs
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs_range = range(1, args.epochs + 1)
    colors = {
        "Baseline (Reproduced)": "#1f77b4",
        "Physics-Informed (lambda=0.001)": "#2ca02c",
        "Physics-Informed (lambda=0.01)": "#d62728",
    }

    for exp in experiments:
        name = exp["name"]
        hist = histories[name]
        c = colors[name]
        axes[0].plot(epochs_range, hist.val_losses, label=name, color=c, lw=2)
        if hist.val_mse_losses is not None:
            axes[1].plot(epochs_range, hist.val_mse_losses, label=name, color=c, lw=2)
        if hist.val_lap_losses is not None:
            axes[2].plot(epochs_range, hist.val_lap_losses, label=name, color=c, lw=2)

    axes[0].set_title(
        r"Total Validation Loss $\mathcal{L}_{\rm total}$",
        fontsize=12,
        fontweight="bold",
    )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Loss Value", fontsize=11)
    axes[0].grid(True, which="both", ls="--", alpha=0.5)
    axes[0].legend(fontsize=9)

    axes[1].set_title(
        r"Validation Field MSE $\mathcal{L}_{\rm MSE}$",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].grid(True, which="both", ls="--", alpha=0.5)
    axes[1].legend(fontsize=9)

    axes[2].set_title(
        r"Validation Laplacian MSE $\mathcal{L}_{\rm Lap}$",
        fontsize=12,
        fontweight="bold",
    )
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Epoch", fontsize=11)
    axes[2].grid(True, which="both", ls="--", alpha=0.5)
    axes[2].legend(fontsize=9)

    plt.tight_layout()
    p_loss = figures_dir / "loss_comparison_laplacian_experiment.png"
    plt.savefig(p_loss, dpi=200)
    plt.close()
    print(f"\nSaved loss curves plot: {p_loss}")

    # Figure 2: Temperature & Laplacian Error Visualizations
    # Pick a high-mode Fourier test sample to visually demonstrate Laplacian fidelity
    sample_traj = splits.test[1]  # sim_fourier_0003
    u0_test = sample_traj.snapshots[0]
    u1_true = sample_traj.snapshots[1]
    t_in = torch.from_numpy(u0_test).unsqueeze(0).unsqueeze(0).float()
    t_ref = torch.from_numpy(u1_true).unsqueeze(0).unsqueeze(0).float()
    lap_true = compute_discrete_laplacian_torch(t_ref, dx, dy).squeeze().numpy()

    models_to_plot = [
        ("Baseline (Rep)", trained_models["Baseline (Reproduced)"]),
        ("Lap (lam=0.001)", trained_models["Physics-Informed (lambda=0.001)"]),
        ("Lap (lam=0.01)", trained_models["Physics-Informed (lambda=0.01)"]),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(16, 11))
    vmax_u = max(float(u1_true.max()), float(u0_test.max()))
    vmin_u = min(float(u1_true.min()), float(u0_test.min()))
    vmax_lap = float(np.max(np.abs(lap_true)))

    for row_idx, (m_label, m_obj) in enumerate(models_to_plot):
        m_obj.eval()
        with torch.no_grad():
            u_p = m_obj(t_in).squeeze().numpy().astype(np.float64)
            t_p = torch.from_numpy(u_p).unsqueeze(0).unsqueeze(0).float()
            lap_p = compute_discrete_laplacian_torch(t_p, dx, dy).squeeze().numpy()

        err_u = np.abs(u1_true - u_p)
        err_lap = np.abs(lap_true - lap_p)

        # Col 0: Predicted Field
        im0 = axes[row_idx, 0].imshow(
            u_p, cmap="inferno", vmin=vmin_u, vmax=vmax_u, origin="lower"
        )
        axes[row_idx, 0].set_title(
            f"{m_label}\nPred $u(t+0.1)$", fontsize=10, fontweight="bold"
        )
        fig.colorbar(im0, ax=axes[row_idx, 0], fraction=0.046, pad=0.04)

        # Col 1: Temperature Error Map
        im1 = axes[row_idx, 1].imshow(err_u, cmap="magma", origin="lower")
        axes[row_idx, 1].set_title(
            f"Temp Error $|u - \\hat{{u}}|$\nMax: {err_u.max():.4f}", fontsize=10
        )
        fig.colorbar(im1, ax=axes[row_idx, 1], fraction=0.046, pad=0.04)

        # Col 2: Predicted Laplacian
        im2 = axes[row_idx, 2].imshow(
            lap_p,
            cmap="coolwarm",
            vmin=-vmax_lap,
            vmax=vmax_lap,
            origin="lower",
        )
        axes[row_idx, 2].set_title(
            r"Pred $\nabla^2 \hat{u}$", fontsize=10, fontweight="bold"
        )
        fig.colorbar(im2, ax=axes[row_idx, 2], fraction=0.046, pad=0.04)

        # Col 3: Laplacian Error Map
        im3 = axes[row_idx, 3].imshow(err_lap, cmap="viridis", origin="lower")
        axes[row_idx, 3].set_title(
            f"Lap Error $|\\nabla^2 u - \\nabla^2 \\hat{{u}}|$\n"
            f"Max: {err_lap.max():.2f}",
            fontsize=10,
        )
        fig.colorbar(im3, ax=axes[row_idx, 3], fraction=0.046, pad=0.04)

        for col in range(4):
            axes[row_idx, col].set_xticks([])
            axes[row_idx, col].set_yticks([])

    title_text = (
        f"Spatial Temperature and Laplacian Fidelity Comparison "
        f"(Test: {sample_traj.simulation_id})"
    )
    plt.suptitle(
        title_text,
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout()
    p_fields = figures_dir / "laplacian_loss_field_comparison.png"
    plt.savefig(p_fields, dpi=200)
    plt.close()
    print(f"Saved field error visualisations: {p_fields}")

    # Figure 3: Rollout Error Curves across Steps
    fig, (ax_ro1, ax_ro2) = plt.subplots(1, 2, figsize=(14, 5))
    times = [0.1 * k for k in range(11)]

    for name in [
        "Baseline (Reproduced)",
        "Physics-Informed (lambda=0.001)",
        "Physics-Informed (lambda=0.01)",
    ]:
        ro = np.array(eval_results[name]["rollout_rel_l2_curve"]) * 100.0
        c = colors[name]
        ax_ro1.plot(times, ro, "o-", label=name, color=c, lw=2.2)

    ax_ro1.set_xlabel("Physical Time $t$ [s]", fontsize=12)
    ax_ro1.set_ylabel("Relative $L_2$ Error [%]", fontsize=12)
    ax_ro1.set_title(
        "10-Step Autoregressive Rollout Error Compounding",
        fontsize=13,
        fontweight="bold",
    )
    ax_ro1.grid(True, which="both", ls="--", alpha=0.5)
    ax_ro1.legend(fontsize=10)

    # Boundary Distance Error Profile Comparison
    distances = np.arange(nx // 2 + 1)
    for name in [
        "Baseline (Reproduced)",
        "Physics-Informed (lambda=0.001)",
        "Physics-Informed (lambda=0.01)",
    ]:
        b_mae = np.array(eval_results[name]["mae_by_distance"])
        c = colors[name]
        ax_ro2.plot(distances, b_mae, "o-", label=name, color=c, lw=2.2)

    ax_ro2.set_xlabel("Distance $d$ from Boundary (Grid Cells)", fontsize=12)
    ax_ro2.set_ylabel("Mean Absolute Error (MAE)", fontsize=12)
    ax_ro2.set_title(
        "Boundary Distance Error Profile vs. Regularization",
        fontsize=13,
        fontweight="bold",
    )
    ax_ro2.grid(True, which="both", ls="--", alpha=0.5)
    ax_ro2.legend(fontsize=10)

    plt.tight_layout()
    p_ro = figures_dir / "rollout_comparison_laplacian_experiment.png"
    plt.savefig(p_ro, dpi=200)
    plt.close()
    print(f"Saved rollout and boundary comparison plots: {p_ro}")

    # 6. Save JSON report
    report_dict = {
        "summary": {
            "configurations": list(eval_results.keys()),
            "best_epochs": {
                name: (histories[name].best_epoch if name in histories else 58)
                for name in eval_results
            },
            "best_val_losses": {
                name: (
                    histories[name].best_val_loss if name in histories else 1.7767e-05
                )
                for name in eval_results
            },
        },
        "test_results": eval_results,
    }

    report_path = models_dir / "laplacian_loss_experiment_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    print(f"Saved experiment report to: {report_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
