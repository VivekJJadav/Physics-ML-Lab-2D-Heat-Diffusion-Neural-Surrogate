"""Milestone 4.4: Normalized Laplacian Supervision Experiment.

Objective:
Investigate whether normalized Laplacian derivative supervision:
    L_total = L_MSE + lambda * L_laplacian_normalized
where:
    L_lap_norm = MSE(nabla_h^2 u_pred, nabla_h^2 u_ref) / (S_lap_sq + eps)
improves spatial derivative fidelity without sacrificing temperature prediction
accuracy or multi-step rollout stability.

Experimental Configurations:
1. Baseline: lambda = 0.0
2. Normalized Laplacian: lambda = 0.01
3. Normalized Laplacian: lambda = 0.05
4. Normalized Laplacian: lambda = 0.1
Evaluated alongside benchmarks (Original M4, Reproduced Baseline, Unnorm PiNN).
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
    compute_laplacian_reference_scale,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Milestone 4.4: Normalized Laplacian Supervision Experiment"
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
    laplacian_scale: float | None,
    train_loader: DataLoader[dict[str, torch.Tensor | str]],
    val_loader: DataLoader[dict[str, torch.Tensor | str]],
    dx: float,
    dy: float,
    epochs: int,
    checkpoint_path: Path,
    seed: int,
) -> tuple[HeatResNet2D, TrainingHistory]:
    """Train HeatResNet2D with normalized Laplacian and CosineAnnealing."""
    set_seed(seed)

    model = HeatResNet2D(in_channels=1, hidden_channels=32, num_blocks=3)
    trainer = SurrogateTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=1e-3,
        weight_decay=1e-4,
        lambda_laplacian=lambda_lap,
        laplacian_scale=laplacian_scale,
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

    # Metric A: Mean of sample relative L2
    rel_l2_sample = float(
        np.mean(
            [
                compute_l2_error(p, r, relative=True)
                for p, r in zip(all_u_pred, all_u_ref, strict=False)
            ]
        )
    )
    # Metric B: Global Frobenius ratio
    sum_sq_diff = sum(np.sum(d**2) for d in diffs)
    sum_sq_ref = sum(np.sum(r**2) for r in all_u_ref)
    rel_l2_frob = float(np.sqrt(sum_sq_diff / max(1e-12, sum_sq_ref)))

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
        sub_sum_sq_diff = sum(np.sum(d**2) for d in sub_d)
        sub_sum_sq_ref = sum(np.sum(r**2) for r in sub_r)
        return {
            "mse": sub_mse,
            "rmse": float(np.sqrt(sub_mse)),
            "relative_l2_sample": float(
                np.mean(
                    [
                        compute_l2_error(p, r, relative=True)
                        for p, r in zip(sub_p, sub_r, strict=False)
                    ]
                )
            ),
            "relative_l2_frob": float(
                np.sqrt(sub_sum_sq_diff / max(1e-12, sub_sum_sq_ref))
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
    lap_rel_l2_sample = float(
        np.mean(
            [
                np.linalg.norm(ld) / max(1e-12, np.linalg.norm(lr))
                for ld, lr in zip(lap_diffs, lap_refs, strict=False)
            ]
        )
    )
    sum_sq_ld = sum(np.sum(ld**2) for ld in lap_diffs)
    sum_sq_lr = sum(np.sum(lr**2) for lr in lap_refs)
    lap_rel_l2_frob = float(np.sqrt(sum_sq_ld / max(1e-12, sum_sq_lr)))

    # 3. Boundary distance profile (MAE for d = 0 to max_d)
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

    # 4. Temperature range & Gaussian Negative Cell Count
    min_all_pred = float(np.min([np.min(p) for p in all_u_pred]))
    max_all_pred = float(np.max([np.max(p) for p in all_u_pred]))

    gauss_preds = [all_u_pred[i] for i in g_idxs]
    min_gauss_val = float(np.min([np.min(p) for p in gauss_preds]))
    max_gauss_val = float(np.max([np.max(p) for p in gauss_preds]))

    # Gaussian interior negative cells:
    # Denominators:
    # Interior nodes per snapshot = (nx - 2) * (ny - 2) = 49 * 49 = 2401
    # Total interior nodes = len(gauss_preds) * 2401 = 20 * 2401 = 48020
    # Total nodes = len(gauss_preds) * nx * ny = 20 * 2601 = 52020
    interior_neg_tol = 0
    interior_neg_strict = 0
    for p in gauss_preds:
        p_int = p[1:-1, 1:-1]
        interior_neg_tol += int(np.sum(p_int < -1e-5))
        interior_neg_strict += int(np.sum(p_int < 0.0))

    total_g_interior_nodes = len(gauss_preds) * (nx - 2) * (ny - 2)
    total_g_nodes = len(gauss_preds) * nx * ny

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
    rollout_rel_l2_curve = ensemble_rollout["mean_relative_l2"]
    final_rollout_l2 = float(rollout_rel_l2_curve[-1])

    # Trajectory-level evolution tracking
    eval_trajs = [
        rollout_evaluator.evaluate_trajectory(traj) for traj in test_trajectories
    ]
    g_eval_trajs = [tr for tr in eval_trajs if tr["generator_type"] == "gaussian"]
    f_eval_trajs = [tr for tr in eval_trajs if tr["generator_type"] == "fourier"]

    # Rollout curves: peak, mean, and energy evolution across steps 0..10
    g_peak_ref = np.mean(
        [np.max(tr["ground_truth"], axis=(1, 2)) for tr in g_eval_trajs], axis=0
    ).tolist()
    g_peak_pred = np.mean(
        [np.max(tr["predicted"], axis=(1, 2)) for tr in g_eval_trajs], axis=0
    ).tolist()
    g_mean_ref = np.mean(
        [np.mean(tr["ground_truth"], axis=(1, 2)) for tr in g_eval_trajs], axis=0
    ).tolist()
    g_mean_pred = np.mean(
        [np.mean(tr["predicted"], axis=(1, 2)) for tr in g_eval_trajs], axis=0
    ).tolist()

    # Energy evolution H(t) = sum(u^2) * dx * dy
    def _energy_steps(trajs: list[dict[str, Any]], key: str) -> list[float]:
        step_energies = []
        for step in range(11):
            e_step = np.mean([np.sum(tr[key][step] ** 2) * dx * dy for tr in trajs])
            step_energies.append(float(e_step))
        return step_energies

    g_energy_ref = _energy_steps(g_eval_trajs, "ground_truth")
    g_energy_pred = _energy_steps(g_eval_trajs, "predicted")

    f_peak_ref = np.mean(
        [np.max(tr["ground_truth"], axis=(1, 2)) for tr in f_eval_trajs], axis=0
    ).tolist()
    f_peak_pred = np.mean(
        [np.max(tr["predicted"], axis=(1, 2)) for tr in f_eval_trajs], axis=0
    ).tolist()
    f_mean_ref = np.mean(
        [np.mean(tr["ground_truth"], axis=(1, 2)) for tr in f_eval_trajs], axis=0
    ).tolist()
    f_mean_pred = np.mean(
        [np.mean(tr["predicted"], axis=(1, 2)) for tr in f_eval_trajs], axis=0
    ).tolist()
    f_energy_ref = _energy_steps(f_eval_trajs, "ground_truth")
    f_energy_pred = _energy_steps(f_eval_trajs, "predicted")

    return {
        "mse": mse,
        "rmse": rmse,
        "relative_l2_sample": rel_l2_sample,
        "relative_l2_frob": rel_l2_frob,
        "max_error": max_err,
        "fourier": fourier_stats,
        "gaussian": gaussian_stats,
        "laplacian_mse": lap_mse,
        "laplacian_rmse": lap_rmse,
        "laplacian_relative_l2_sample": lap_rel_l2_sample,
        "laplacian_relative_l2_frob": lap_rel_l2_frob,
        "mae_by_distance": mae_by_d.tolist(),
        "d1_mae": float(mae_by_d[1]),
        "d2_mae": float(mae_by_d[2]),
        "d15_mae": float(mae_by_d[15]),
        "min_all_pred": min_all_pred,
        "max_all_pred": max_all_pred,
        "gaussian_min_temp": min_gauss_val,
        "gaussian_max_temp": max_gauss_val,
        "gaussian_neg_cells_strict": interior_neg_strict,
        "gaussian_neg_cells_tol": interior_neg_tol,
        "gaussian_neg_ratio_interior": interior_neg_tol / total_g_interior_nodes,
        "gaussian_neg_ratio_total": interior_neg_tol / total_g_nodes,
        "energy_dissipation_mae": energy_diss_mae,
        "rollout_rel_l2_curve": rollout_rel_l2_curve,
        "final_rollout_rel_l2": final_rollout_l2,
        "rollout_amplification": float(final_rollout_l2 / max(1e-12, rel_l2_sample)),
        "gaussian_peak_ref": g_peak_ref,
        "gaussian_peak_pred": g_peak_pred,
        "gaussian_mean_ref": g_mean_ref,
        "gaussian_mean_pred": g_mean_pred,
        "gaussian_energy_ref": g_energy_ref,
        "gaussian_energy_pred": g_energy_pred,
        "fourier_peak_ref": f_peak_ref,
        "fourier_peak_pred": f_peak_pred,
        "fourier_mean_ref": f_mean_ref,
        "fourier_mean_pred": f_mean_pred,
        "fourier_energy_ref": f_energy_ref,
        "fourier_energy_pred": f_energy_pred,
    }


def main() -> None:
    args = parse_args()
    print("=" * 80)
    print("MILESTONE 4.4: NORMALIZED LAPLACIAN SUPERVISION EXPERIMENT")
    print("=" * 80)

    repo_root = Path(__file__).resolve().parent.parent
    dataset_dir = repo_root / args.dataset_dir
    models_dir = repo_root / args.models_dir
    figures_dir = repo_root / args.figures_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load dataset splits
    splits = load_dataset(dataset_dir)
    config = splits.train[0].config
    dx, dy = config.dx, config.dy
    nx, ny = config.grid_resolution_x, config.grid_resolution_y

    # 2. Compute Normalization Scale S_lap_squared using training targets only
    train_targets_list = []
    for traj in splits.train:
        for t in range(len(traj.timestamps) - 1):
            train_targets_list.append(traj.snapshots[t + 1])

    train_targets_tensor = (
        torch.from_numpy(np.array(train_targets_list)).unsqueeze(1).float()
    )
    s_lap_squared = compute_laplacian_reference_scale(train_targets_tensor, dx, dy)
    s_lap_rms = float(np.sqrt(s_lap_squared))

    scale_metadata = {
        "dataset_split": "train",
        "num_simulations": len(splits.train),
        "num_target_snapshots": len(train_targets_list),
        "grid_resolution": [nx, ny],
        "interior_grid_resolution": [nx - 2, ny - 2],
        "total_interior_evaluations": len(train_targets_list) * (nx - 2) * (ny - 2),
        "dx": dx,
        "dy": dy,
        "s_lap_squared": s_lap_squared,
        "s_lap_rms": s_lap_rms,
        "provenance": (
            "Computed exclusively on training targets u(t+0.1) using 5-point discrete "
            "Laplacian stencil on interior cells [1:-1, 1:-1], "
            "strictly excluding boundaries."
        ),
    }

    print("\n--- Fixed Laplacian Normalization Scale Provenance ---")
    for k, v in scale_metadata.items():
        print(f"  {k}: {v}")

    # Prepare DataLoaders
    train_ds = HeatDiffusionDataset(splits.train, mode="step")
    val_ds = HeatDiffusionDataset(splits.val, mode="step")
    test_trajectories = splits.test

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # 3. Experimental Configurations
    experiments = [
        {
            "name": "Baseline (Reproduced, lambda=0)",
            "lambda_lap": 0.0,
            "laplacian_scale": None,
            "checkpoint": models_dir / "surrogate_baseline_norm_exp.pt",
        },
        {
            "name": "Norm-Laplacian (lambda=0.01)",
            "lambda_lap": 0.01,
            "laplacian_scale": s_lap_squared,
            "checkpoint": models_dir / "surrogate_norm_laplacian_lambda_0.01.pt",
        },
        {
            "name": "Norm-Laplacian (lambda=0.05)",
            "lambda_lap": 0.05,
            "laplacian_scale": s_lap_squared,
            "checkpoint": models_dir / "surrogate_norm_laplacian_lambda_0.05.pt",
        },
        {
            "name": "Norm-Laplacian (lambda=0.1)",
            "lambda_lap": 0.1,
            "laplacian_scale": s_lap_squared,
            "checkpoint": models_dir / "surrogate_norm_laplacian_lambda_0.1.pt",
        },
    ]

    histories: dict[str, TrainingHistory] = {}
    eval_results: dict[str, dict[str, Any]] = {}

    for exp in experiments:
        name = exp["name"]
        ckpt_path = exp["checkpoint"]
        lambda_lap = exp["lambda_lap"]
        lap_scale = exp["laplacian_scale"]

        print(f"\n>>> Training Configuration: {name} (lambda={lambda_lap})")
        model, history = train_model_configuration(
            lambda_lap=lambda_lap,
            laplacian_scale=lap_scale,
            train_loader=train_loader,
            val_loader=val_loader,
            dx=dx,
            dy=dy,
            epochs=args.epochs,
            checkpoint_path=ckpt_path,
            seed=args.seed,
        )
        histories[name] = history

        print(
            f"    Best Epoch: {history.best_epoch} | "
            f"Best Val Objective: {history.best_val_loss:.6e}"
        )

        # Full evaluation
        eval_res = evaluate_model_full(
            model=model,
            test_trajectories=test_trajectories,
            dx=dx,
            dy=dy,
            nx=nx,
            ny=ny,
        )
        eval_results[name] = eval_res
        print(
            f"    Test MSE: {eval_res['mse']:.6e} | "
            f"Test Rel L2: {eval_res['relative_l2_sample'] * 100:.3f}% | "
            f"Lap Rel L2: {eval_res['laplacian_relative_l2_sample'] * 100:.3f}% | "
            f"Rollout Rel L2: {eval_res['final_rollout_rel_l2'] * 100:.3f}%"
        )

    # Also benchmark existing checkpoints
    benchmark_checkpoints = {
        "Original Baseline (M4)": models_dir / "best_surrogate.pt",
        "Baseline (Reproduced M4.2)": models_dir / "surrogate_baseline_reproduced.pt",
        "Unnormalized PiNN (lambda=0.001)": models_dir
        / "surrogate_laplacian_lambda_0.001.pt",
        "Unnormalized PiNN (lambda=0.01)": models_dir
        / "surrogate_laplacian_lambda_0.01.pt",
    }
    for b_name, b_path in benchmark_checkpoints.items():
        if b_path.exists():
            print(f"\n>>> Benchmarking Historical Checkpoint: {b_name}")
            ckpt = torch.load(b_path, map_location="cpu")
            b_model = HeatResNet2D(in_channels=1, hidden_channels=32, num_blocks=3)
            b_model.load_state_dict(ckpt["model_state_dict"])
            b_eval = evaluate_model_full(
                model=b_model,
                test_trajectories=test_trajectories,
                dx=dx,
                dy=dy,
                nx=nx,
                ny=ny,
            )
            eval_results[b_name] = b_eval

    # 4. Comparative Printout Table
    print("\n" + "=" * 110)
    print("MILESTONE 4.4: TEST METRICS COMPARISON TABLE")
    print("=" * 110)
    hdr = (
        f"{'Configuration':<34} | {'Field MSE':<12} | {'Rel L2 [%]':<11} | "
        f"{'Lap RMSE':<10} | {'Lap Rel [%]':<11} | {'Rollout [%]':<11} | "
        f"{'G-Neg Cells':<11}"
    )
    print(hdr)
    print("-" * len(hdr))
    all_report_names = [exp["name"] for exp in experiments] + list(
        benchmark_checkpoints.keys()
    )
    for name in all_report_names:
        if name in eval_results:
            res = eval_results[name]
            print(
                f"{name:<34} | {res['mse']:<12.6e} | "
                f"{res['relative_l2_sample'] * 100:<11.3f} | "
                f"{res['laplacian_rmse']:<10.4f} | "
                f"{res['laplacian_relative_l2_sample'] * 100:<11.3f} | "
                f"{res['final_rollout_rel_l2'] * 100:<11.3f} | "
                f"{res['gaussian_neg_cells_tol']:<11}"
            )

    # 5. Diagnostic Figures Generation
    epochs_range = range(1, args.epochs + 1)
    colors = {
        "Baseline (Reproduced, lambda=0)": "#1f77b4",
        "Norm-Laplacian (lambda=0.01)": "#2ca02c",
        "Norm-Laplacian (lambda=0.05)": "#ff7f0e",
        "Norm-Laplacian (lambda=0.1)": "#d62728",
    }

    # Figure 1: Training & Validation Loss Curves across Epochs
    _fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Total Validation Loss
    for exp in experiments:
        name = exp["name"]
        hist = histories[name]
        c = colors[name]
        axes[0, 0].plot(epochs_range, hist.val_losses, label=name, color=c, lw=2)
    axes[0, 0].set_title(
        r"Total Validation Loss $\mathcal{L}_{\rm total}$",
        fontsize=12,
        fontweight="bold",
    )
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("Epoch", fontsize=10)
    axes[0, 0].set_ylabel("Loss", fontsize=10)
    axes[0, 0].grid(True, which="both", ls="--", alpha=0.5)
    axes[0, 0].legend(fontsize=9)

    # Validation Field MSE
    for exp in experiments:
        name = exp["name"]
        hist = histories[name]
        c = colors[name]
        if hist.val_mse_losses:
            axes[0, 1].plot(
                epochs_range, hist.val_mse_losses, label=name, color=c, lw=2
            )
    axes[0, 1].set_title(
        r"Validation Field MSE $\mathcal{L}_{\rm MSE}$", fontsize=12, fontweight="bold"
    )
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlabel("Epoch", fontsize=10)
    axes[0, 1].grid(True, which="both", ls="--", alpha=0.5)
    axes[0, 1].legend(fontsize=9)

    # Validation Normalized Laplacian Loss
    for exp in experiments:
        name = exp["name"]
        hist = histories[name]
        c = colors[name]
        if hist.val_lap_norm_losses:
            axes[1, 0].plot(
                epochs_range, hist.val_lap_norm_losses, label=name, color=c, lw=2
            )
    axes[1, 0].set_title(
        r"Validation Normalized Laplacian Loss $\mathcal{L}_{\rm lap\_norm}$",
        fontsize=12,
        fontweight="bold",
    )
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel("Epoch", fontsize=10)
    axes[1, 0].grid(True, which="both", ls="--", alpha=0.5)
    axes[1, 0].legend(fontsize=9)

    # Weighted Laplacian Contribution: lambda * L_lap_norm
    for exp in experiments:
        name = exp["name"]
        hist = histories[name]
        c = colors[name]
        if hist.weighted_val_lap_losses:
            axes[1, 1].plot(
                epochs_range, hist.weighted_val_lap_losses, label=name, color=c, lw=2
            )
    axes[1, 1].set_title(
        r"Weighted Laplacian Term $\lambda \cdot \mathcal{L}_{\rm lap\_norm}$",
        fontsize=12,
        fontweight="bold",
    )
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("Epoch", fontsize=10)
    axes[1, 1].grid(True, which="both", ls="--", alpha=0.5)
    axes[1, 1].legend(fontsize=9)

    plt.tight_layout()
    fig1_path = figures_dir / "loss_curves_normalized_laplacian.png"
    plt.savefig(fig1_path, dpi=200)
    plt.close()
    print(f"Saved: {fig1_path}")

    # Figure 2: Rollout Comparison (Relative L2, Peak, Mean, and Energy decay)
    _fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    steps = np.arange(11)

    # Rollout Relative L2 Error Progression
    for exp in experiments:
        name = exp["name"]
        res = eval_results[name]
        c = colors[name]
        axes2[0, 0].plot(
            steps,
            np.array(res["rollout_rel_l2_curve"]) * 100,
            label=name,
            color=c,
            lw=2,
            marker="o",
            ms=4,
        )
    # Add unnormalized benchmark for contrast
    if "Unnormalized PiNN (lambda=0.001)" in eval_results:
        res_u = eval_results["Unnormalized PiNN (lambda=0.001)"]
        axes2[0, 0].plot(
            steps,
            np.array(res_u["rollout_rel_l2_curve"]) * 100,
            label="Unnorm PiNN (lambda=0.001)",
            color="gray",
            ls="--",
            lw=1.5,
        )

    axes2[0, 0].set_title(
        "10-Step Autoregressive Rollout Relative $L_2$ [%]",
        fontsize=12,
        fontweight="bold",
    )
    axes2[0, 0].set_xlabel("Rollout Step (k)", fontsize=10)
    axes2[0, 0].set_ylabel("Relative $L_2$ Error [%]", fontsize=10)
    axes2[0, 0].grid(True, ls="--", alpha=0.5)
    axes2[0, 0].legend(fontsize=8)

    # Peak Temperature Evolution (Gaussian)
    ref_g_peak = eval_results["Baseline (Reproduced, lambda=0)"]["gaussian_peak_ref"]
    axes2[0, 1].plot(
        steps, ref_g_peak, label="Ground Truth", color="black", lw=2.5, ls="--"
    )
    for exp in experiments:
        name = exp["name"]
        res = eval_results[name]
        c = colors[name]
        axes2[0, 1].plot(steps, res["gaussian_peak_pred"], label=name, color=c, lw=1.8)
    axes2[0, 1].set_title(
        "Gaussian Peak Temperature Decay $u_{\\rm max}(t)$",
        fontsize=12,
        fontweight="bold",
    )
    axes2[0, 1].set_xlabel("Rollout Step (k)", fontsize=10)
    axes2[0, 1].set_ylabel("Peak Temperature", fontsize=10)
    axes2[0, 1].grid(True, ls="--", alpha=0.5)
    axes2[0, 1].legend(fontsize=8)

    # Mean Temperature Evolution (Gaussian)
    ref_g_mean = eval_results["Baseline (Reproduced, lambda=0)"]["gaussian_mean_ref"]
    axes2[1, 0].plot(
        steps, ref_g_mean, label="Ground Truth", color="black", lw=2.5, ls="--"
    )
    for exp in experiments:
        name = exp["name"]
        res = eval_results[name]
        c = colors[name]
        axes2[1, 0].plot(steps, res["gaussian_mean_pred"], label=name, color=c, lw=1.8)
    axes2[1, 0].set_title(
        "Gaussian Mean Temperature Evolution $\\bar{u}(t)$",
        fontsize=12,
        fontweight="bold",
    )
    axes2[1, 0].set_xlabel("Rollout Step (k)", fontsize=10)
    axes2[1, 0].set_ylabel("Mean Temperature", fontsize=10)
    axes2[1, 0].grid(True, ls="--", alpha=0.5)
    axes2[1, 0].legend(fontsize=8)

    # Thermal Energy Evolution H(t)
    ref_g_energy = eval_results["Baseline (Reproduced, lambda=0)"][
        "gaussian_energy_ref"
    ]
    axes2[1, 1].plot(
        steps, ref_g_energy, label="Ground Truth", color="black", lw=2.5, ls="--"
    )
    for exp in experiments:
        name = exp["name"]
        res = eval_results[name]
        c = colors[name]
        axes2[1, 1].plot(
            steps, res["gaussian_energy_pred"], label=name, color=c, lw=1.8
        )
    axes2[1, 1].set_title(
        "Thermal Energy Dissipation $H(t) = \\int u^2 dA$",
        fontsize=12,
        fontweight="bold",
    )
    axes2[1, 1].set_xlabel("Rollout Step (k)", fontsize=10)
    axes2[1, 1].set_ylabel("Energy Proxy $H(t)$", fontsize=10)
    axes2[1, 1].grid(True, ls="--", alpha=0.5)
    axes2[1, 1].legend(fontsize=8)

    plt.tight_layout()
    fig2_path = figures_dir / "rollout_comparison_normalized_laplacian.png"
    plt.savefig(fig2_path, dpi=200)
    plt.close()
    print(f"Saved: {fig2_path}")

    # Figure 3: Spatial Field and Error Maps Comparison for a Gaussian Test Simulation
    target_sim = next(s for s in test_trajectories if s.generator_type == "gaussian")
    eval_model_baseline = HeatResNet2D(in_channels=1, hidden_channels=32, num_blocks=3)
    eval_model_baseline.load_state_dict(
        torch.load(experiments[0]["checkpoint"])["model_state_dict"]
    )
    eval_model_norm1 = HeatResNet2D(in_channels=1, hidden_channels=32, num_blocks=3)
    eval_model_norm1.load_state_dict(
        torch.load(experiments[1]["checkpoint"])["model_state_dict"]
    )
    eval_model_norm3 = HeatResNet2D(in_channels=1, hidden_channels=32, num_blocks=3)
    eval_model_norm3.load_state_dict(
        torch.load(experiments[3]["checkpoint"])["model_state_dict"]
    )

    r_base = RolloutEvaluator(eval_model_baseline, device="cpu").predict_rollout(
        target_sim.initial_state, 10
    )
    r_n1 = RolloutEvaluator(eval_model_norm1, device="cpu").predict_rollout(
        target_sim.initial_state, 10
    )
    r_n3 = RolloutEvaluator(eval_model_norm3, device="cpu").predict_rollout(
        target_sim.initial_state, 10
    )
    gt = target_sim.snapshots

    _fig3, axes3 = plt.subplots(3, 4, figsize=(16, 11))
    step_idx = 10  # final rollout state t = 1.0s

    im0 = axes3[0, 0].imshow(gt[step_idx], cmap="inferno", origin="lower")
    axes3[0, 0].set_title(
        f"Reference Ground Truth (t=1.0s)\n[{target_sim.simulation_id}]",
        fontsize=10,
        fontweight="bold",
    )
    plt.colorbar(im0, ax=axes3[0, 0], fraction=0.046)

    im1 = axes3[0, 1].imshow(r_base[step_idx], cmap="inferno", origin="lower")
    axes3[0, 1].set_title(
        "Baseline (lambda=0)\nPrediction (t=1.0s)", fontsize=10, fontweight="bold"
    )
    plt.colorbar(im1, ax=axes3[0, 1], fraction=0.046)

    im2 = axes3[0, 2].imshow(r_n1[step_idx], cmap="inferno", origin="lower")
    axes3[0, 2].set_title(
        "Norm-Lap (lambda=0.01)\nPrediction (t=1.0s)", fontsize=10, fontweight="bold"
    )
    plt.colorbar(im2, ax=axes3[0, 2], fraction=0.046)

    im3 = axes3[0, 3].imshow(r_n3[step_idx], cmap="inferno", origin="lower")
    axes3[0, 3].set_title(
        "Norm-Lap (lambda=0.1)\nPrediction (t=1.0s)", fontsize=10, fontweight="bold"
    )
    plt.colorbar(im3, ax=axes3[0, 3], fraction=0.046)

    # Row 2: Signed Error Maps (Pred - Ref)
    err_base = r_base[step_idx] - gt[step_idx]
    err_n1 = r_n1[step_idx] - gt[step_idx]
    err_n3 = r_n3[step_idx] - gt[step_idx]
    v_max = max(
        np.max(np.abs(err_base)), np.max(np.abs(err_n1)), np.max(np.abs(err_n3))
    )

    axes3[1, 0].axis("off")
    im_e1 = axes3[1, 1].imshow(
        err_base, cmap="coolwarm", vmin=-v_max, vmax=v_max, origin="lower"
    )
    axes3[1, 1].set_title("Baseline Error (Pred - Ref)", fontsize=10)
    plt.colorbar(im_e1, ax=axes3[1, 1], fraction=0.046)

    im_e2 = axes3[1, 2].imshow(
        err_n1, cmap="coolwarm", vmin=-v_max, vmax=v_max, origin="lower"
    )
    axes3[1, 2].set_title("Norm-Lap (0.01) Error", fontsize=10)
    plt.colorbar(im_e2, ax=axes3[1, 2], fraction=0.046)

    im_e3 = axes3[1, 3].imshow(
        err_n3, cmap="coolwarm", vmin=-v_max, vmax=v_max, origin="lower"
    )
    axes3[1, 3].set_title("Norm-Lap (0.1) Error", fontsize=10)
    plt.colorbar(im_e3, ax=axes3[1, 3], fraction=0.046)

    # Row 3: Discrete Spatial Laplacian Fields
    gt_lap = (
        compute_discrete_laplacian_torch(
            torch.from_numpy(gt[step_idx]).unsqueeze(0).unsqueeze(0).float(), dx, dy
        )
        .squeeze()
        .numpy()
    )
    base_lap = (
        compute_discrete_laplacian_torch(
            torch.from_numpy(r_base[step_idx]).unsqueeze(0).unsqueeze(0).float(), dx, dy
        )
        .squeeze()
        .numpy()
    )
    n1_lap = (
        compute_discrete_laplacian_torch(
            torch.from_numpy(r_n1[step_idx]).unsqueeze(0).unsqueeze(0).float(), dx, dy
        )
        .squeeze()
        .numpy()
    )
    n3_lap = (
        compute_discrete_laplacian_torch(
            torch.from_numpy(r_n3[step_idx]).unsqueeze(0).unsqueeze(0).float(), dx, dy
        )
        .squeeze()
        .numpy()
    )

    im_l0 = axes3[2, 0].imshow(gt_lap, cmap="viridis", origin="lower")
    axes3[2, 0].set_title("Reference Laplacian $\\nabla_h^2 u$", fontsize=10)
    plt.colorbar(im_l0, ax=axes3[2, 0], fraction=0.046)

    im_l1 = axes3[2, 1].imshow(base_lap, cmap="viridis", origin="lower")
    axes3[2, 1].set_title("Baseline Laplacian", fontsize=10)
    plt.colorbar(im_l1, ax=axes3[2, 1], fraction=0.046)

    im_l2 = axes3[2, 2].imshow(n1_lap, cmap="viridis", origin="lower")
    axes3[2, 2].set_title("Norm-Lap (0.01) Laplacian", fontsize=10)
    plt.colorbar(im_l2, ax=axes3[2, 2], fraction=0.046)

    im_l3 = axes3[2, 3].imshow(n3_lap, cmap="viridis", origin="lower")
    axes3[2, 3].set_title("Norm-Lap (0.1) Laplacian", fontsize=10)
    plt.colorbar(im_l3, ax=axes3[2, 3], fraction=0.046)

    plt.tight_layout()
    fig3_path = figures_dir / "field_comparison_normalized_laplacian.png"
    plt.savefig(fig3_path, dpi=200)
    plt.close()
    print(f"Saved: {fig3_path}")

    # 6. Save Complete Experiment JSON Report
    report_data = {
        "metadata": {
            "milestone": "4.4",
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "scale_metadata": scale_metadata,
        },
        "histories": {name: hist.to_dict() for name, hist in histories.items()},
        "eval_results": eval_results,
    }
    report_path = models_dir / "normalized_laplacian_experiment_report.json"
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"\nSaved complete experiment report: {report_path}")
    print("Milestone 4.4 Experiment Completed Successfully.")


if __name__ == "__main__":
    main()
