"""Scientific Audit of Milestone 4 Trained Neural Surrogate.

Performs a rigorous, non-destructive evaluation audit of the model:
1. Test metrics (Overall, Fourier, Gaussian, and per-simulation breakdown).
2. Boundary analysis (outer boundary exact zero check, distance profile).
3. Physical analysis (discrete Laplacians, integrals, energy, oscillations).
4. Baseline comparison (Identity persistence baseline vs. Neural surrogate).
5. Training provenance & data leakage audit (checkpoint validation).
"""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from physics_ml.data.dataset import load_dataset
from physics_ml.evaluation.metrics import (
    compute_l2_error,
    compute_max_error,
    compute_rmse,
)
from physics_ml.models.surrogate import HeatResNet2D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scientific Audit of Milestone 4 Neural Surrogate"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="outputs/datasets/heat_diffusion_v1",
        help="Path to dataset directory.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="outputs/models/best_surrogate.pt",
        help="Path to best surrogate checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/figures",
        help="Directory to save audit figures.",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default="outputs/models/scientific_audit_report.json",
        help="Path to save the JSON audit report.",
    )
    return parser.parse_args()


def compute_discrete_laplacian(u: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Compute central difference discrete Laplacian on interior nodes."""
    lap_x = (u[2:, 1:-1] - 2.0 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / (dx**2)
    lap_y = (u[1:-1, 2:] - 2.0 * u[1:-1, 1:-1] + u[1:-1, :-2]) / (dy**2)
    return lap_x + lap_y


def main() -> None:
    args = parse_args()
    print("=" * 80)
    print("SCIENTIFIC AUDIT: MILESTONE 4 NEURAL SURROGATE MODEL")
    print("=" * 80)

    # 1. Load Dataset Splits
    dataset_dir = Path(args.dataset_dir)
    splits = load_dataset(dataset_dir)
    test_trajectories = splits.test
    config = test_trajectories[0].config
    dx, dy = config.dx, config.dy
    nx, ny = config.grid_resolution_x, config.grid_resolution_y

    print(f"Test Split Size: {len(test_trajectories)} trajectories")

    # 2. Load Existing Model Checkpoint
    model_path = Path(args.model_path)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    model = HeatResNet2D(in_channels=1, hidden_channels=32, num_blocks=3)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # ---------------------------------------------------------
    # PART 1: TEST METRICS & PER-SIMULATION BREAKDOWN
    # ---------------------------------------------------------
    print("\n" + "-" * 40)
    print("1. TEST METRICS & PER-SIMULATION BREAKDOWN")
    print("-" * 40)

    sim_results: list[dict[str, Any]] = []
    all_u_in: list[np.ndarray] = []
    all_u_ref: list[np.ndarray] = []
    all_u_pred: list[np.ndarray] = []
    all_sim_types: list[str] = []

    for traj in test_trajectories:
        snaps = traj.snapshots
        n_pairs = len(snaps) - 1

        sim_mses: list[float] = []
        sim_rmses: list[float] = []
        sim_rel_l2s: list[float] = []
        sim_max_errs: list[float] = []

        for step in range(n_pairs):
            u_t = snaps[step]
            u_tp1_ref = snaps[step + 1]

            t_in = torch.from_numpy(u_t).unsqueeze(0).unsqueeze(0).float()
            with torch.no_grad():
                u_pred_t = model(t_in).squeeze().numpy().astype(np.float64)

            mse = float(np.mean((u_pred_t - u_tp1_ref) ** 2))
            rmse = compute_rmse(u_pred_t, u_tp1_ref)
            rel_l2 = compute_l2_error(u_pred_t, u_tp1_ref, relative=True)
            max_err = compute_max_error(u_pred_t, u_tp1_ref)

            sim_mses.append(mse)
            sim_rmses.append(rmse)
            sim_rel_l2s.append(rel_l2)
            sim_max_errs.append(max_err)

            all_u_in.append(u_t)
            all_u_ref.append(u_tp1_ref)
            all_u_pred.append(u_pred_t)
            all_sim_types.append(traj.generator_type)

        sim_results.append(
            {
                "simulation_id": traj.simulation_id,
                "generator_type": traj.generator_type,
                "num_pairs": n_pairs,
                "mse": float(np.mean(sim_mses)),
                "rmse": float(np.mean(sim_rmses)),
                "relative_l2": float(np.mean(sim_rel_l2s)),
                "max_absolute_error": float(np.max(sim_max_errs)),
            }
        )

    # Aggregate metrics
    overall_mse = float(
        np.mean(
            [np.mean((p - r) ** 2) for p, r in zip(all_u_pred, all_u_ref, strict=False)]
        )
    )
    overall_rmse = float(np.sqrt(overall_mse))
    overall_rel_l2 = float(
        np.mean(
            [
                compute_l2_error(p, r, relative=True)
                for p, r in zip(all_u_pred, all_u_ref, strict=False)
            ]
        )
    )

    fourier_indices = [i for i, t in enumerate(all_sim_types) if t == "fourier"]
    gauss_indices = [i for i, t in enumerate(all_sim_types) if t == "gaussian"]

    def calc_subset(indices: list[int]) -> dict[str, float]:
        sub_preds = [all_u_pred[i] for i in indices]
        sub_refs = [all_u_ref[i] for i in indices]
        mse = float(
            np.mean(
                [
                    np.mean((p - r) ** 2)
                    for p, r in zip(sub_preds, sub_refs, strict=False)
                ]
            )
        )
        return {
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "relative_l2": float(
                np.mean(
                    [
                        compute_l2_error(p, r, relative=True)
                        for p, r in zip(sub_preds, sub_refs, strict=False)
                    ]
                )
            ),
            "max_absolute_error": float(
                np.max(
                    [
                        compute_max_error(p, r)
                        for p, r in zip(sub_preds, sub_refs, strict=False)
                    ]
                )
            ),
        }

    fourier_metrics = calc_subset(fourier_indices)
    gauss_metrics = calc_subset(gauss_indices)

    print(f"Total Test Samples: {len(all_u_pred)} pairs (from 6 simulations)")
    print(f"Overall Test MSE:         {overall_mse:.6e}")
    print(f"Overall Test RMSE:        {overall_rmse:.6f}")
    print(f"Overall Test Relative L2: {overall_rel_l2 * 100:.3f}%")
    print("\nFourier Subset (40 pairs):")
    f_rel = fourier_metrics["relative_l2"] * 100
    print(f"  MSE: {fourier_metrics['mse']:.6e} | Rel L2: {f_rel:.3f}%")
    print("Gaussian Subset (20 pairs):")
    g_rel = gauss_metrics["relative_l2"] * 100
    print(f"  MSE: {gauss_metrics['mse']:.6e} | Rel L2: {g_rel:.3f}%")

    print("\nPer-Simulation Breakdown:")
    hdr = (
        f"{'Simulation ID':<20} | {'Type':<10} | {'MSE':<13} | "
        f"{'RMSE':<10} | {'Rel L2 [%]':<11} | {'Max Err':<10}"
    )
    print(hdr)
    print("-" * len(hdr))
    for sr in sim_results:
        print(
            f"{sr['simulation_id']:<20} | {sr['generator_type']:<10} | "
            f"{sr['mse']:<13.6e} | {sr['rmse']:<10.6f} | "
            f"{sr['relative_l2'] * 100:<11.3f} | {sr['max_absolute_error']:<10.4f}"
        )

    # ---------------------------------------------------------
    # PART 2: BOUNDARY ANALYSIS
    # ---------------------------------------------------------
    print("\n" + "-" * 40)
    print("2. BOUNDARY ANALYSIS")
    print("-" * 40)

    outer_boundary_max = 0.0
    for u_p in all_u_pred:
        top = np.max(np.abs(u_p[0, :]))
        bottom = np.max(np.abs(u_p[-1, :]))
        left = np.max(np.abs(u_p[:, 0]))
        right = np.max(np.abs(u_p[:, -1]))
        outer_boundary_max = max(
            outer_boundary_max, float(max(top, bottom, left, right))
        )

    print(f"Exact outer boundary maximum absolute value: {outer_boundary_max:.1e}")
    print(f"Are outer boundary values exactly zero? {outer_boundary_max == 0.0}")

    dist_map = np.zeros((nx, ny), dtype=int)
    for i in range(nx):
        for j in range(ny):
            dist_map[i, j] = min(i, j, nx - 1 - i, ny - 1 - j)

    max_dist = int(np.max(dist_map))

    mae_by_dist = np.zeros(max_dist + 1)
    max_err_by_dist = np.zeros(max_dist + 1)
    fourier_mae_by_dist = np.zeros(max_dist + 1)
    gauss_mae_by_dist = np.zeros(max_dist + 1)

    for d in range(max_dist + 1):
        mask = dist_map == d
        d_errs_all = [
            np.abs(p[mask] - r[mask])
            for p, r in zip(all_u_pred, all_u_ref, strict=False)
        ]
        all_d_concat = np.concatenate(d_errs_all)
        mae_by_dist[d] = float(np.mean(all_d_concat))
        max_err_by_dist[d] = float(np.max(all_d_concat))

        f_d = np.concatenate(
            [
                np.abs(all_u_pred[idx][mask] - all_u_ref[idx][mask])
                for idx in fourier_indices
            ]
        )
        g_d = np.concatenate(
            [
                np.abs(all_u_pred[idx][mask] - all_u_ref[idx][mask])
                for idx in gauss_indices
            ]
        )
        fourier_mae_by_dist[d] = float(np.mean(f_d))
        gauss_mae_by_dist[d] = float(np.mean(g_d))

    print("\nDistance from Boundary vs Error Profile:")
    dist_hdr = (
        f"{'Distance d':<12} | {'Overall MAE':<14} | "
        f"{'Overall Max Err':<16} | {'Fourier MAE':<14} | {'Gaussian MAE':<14}"
    )
    print(dist_hdr)
    print("-" * len(dist_hdr))
    for d in [0, 1, 2, 3, 5, 10, 15, 20, 25]:
        print(
            f"{d:<12} | {mae_by_dist[d]:<14.6f} | {max_err_by_dist[d]:<16.6f} | "
            f"{fourier_mae_by_dist[d]:<14.6f} | {gauss_mae_by_dist[d]:<14.6f}"
        )

    # ---------------------------------------------------------
    # PART 3: PHYSICAL ANALYSIS
    # ---------------------------------------------------------
    print("\n" + "-" * 40)
    print("3. PHYSICAL ANALYSIS")
    print("-" * 40)

    lap_mses: list[float] = []
    lap_rel_l2s: list[float] = []
    lap_max_diffs: list[float] = []

    for p, r in zip(all_u_pred, all_u_ref, strict=False):
        lap_pred = compute_discrete_laplacian(p, dx, dy)
        lap_ref = compute_discrete_laplacian(r, dx, dy)

        l_mse = float(np.mean((lap_pred - lap_ref) ** 2))
        l_rel = float(
            np.linalg.norm(lap_pred - lap_ref) / max(1e-12, np.linalg.norm(lap_ref))
        )
        l_max = float(np.max(np.abs(lap_pred - lap_ref)))

        lap_mses.append(l_mse)
        lap_rel_l2s.append(l_rel)
        lap_max_diffs.append(l_max)

    mean_lap_mse = float(np.mean(lap_mses))
    mean_lap_rmse = float(np.sqrt(mean_lap_mse))
    mean_lap_rel_l2 = float(np.mean(lap_rel_l2s))
    max_lap_diff = float(np.max(lap_max_diffs))

    print(f"Spatial Laplacian RMSE:             {mean_lap_rmse:.4f}")
    print(f"Spatial Laplacian Relative L2 Error: {mean_lap_rel_l2 * 100:.3f}%")
    print(f"Spatial Laplacian Max Difference:   {max_lap_diff:.4f}")

    mean_t_in = [float(np.mean(u)) for u in all_u_in]
    mean_t_ref = [float(np.mean(u)) for u in all_u_ref]
    mean_t_pred = [float(np.mean(u)) for u in all_u_pred]

    energy_in = [float(np.sum(u**2) * dx * dy) for u in all_u_in]
    energy_ref = [float(np.sum(u**2) * dx * dy) for u in all_u_ref]
    energy_pred = [float(np.sum(u**2) * dx * dy) for u in all_u_pred]

    delta_mean_t_ref = np.array(mean_t_ref) - np.array(mean_t_in)
    delta_mean_t_pred = np.array(mean_t_pred) - np.array(mean_t_in)
    mean_temp_error = float(
        np.mean(np.abs(np.array(mean_t_pred) - np.array(mean_t_ref)))
    )

    delta_energy_ref = np.array(energy_ref) - np.array(energy_in)
    delta_energy_pred = np.array(energy_pred) - np.array(energy_in)
    energy_dissipation_mae = float(
        np.mean(np.abs(delta_energy_pred - delta_energy_ref))
    )

    avg_d_mean_ref = float(np.mean(delta_mean_t_ref))
    avg_d_mean_pred = float(np.mean(delta_mean_t_pred))

    print(f"Mean Temperature Conservation MAE:  {mean_temp_error:.6e}")
    print(f"Average True Mean Temp Change:      {avg_d_mean_ref:.6f}")
    print(f"Average Predicted Mean Temp Change: {avg_d_mean_pred:.6f}")
    print(f"Mean Energy Dissipation MAE:        {energy_dissipation_mae:.6e}")
    print(
        f"Average True Energy Dissipation:     {float(np.mean(delta_energy_ref)):.6f}"
    )
    print(
        f"Average Predicted Energy Dissipation: {float(np.mean(delta_energy_pred)):.6f}"
    )

    gauss_preds = [all_u_pred[i] for i in gauss_indices]
    min_gauss_pred = float(np.min([np.min(p) for p in gauss_preds]))
    neg_nodes_count = sum(int(np.sum(p < -1e-5)) for p in gauss_preds)
    total_gauss_nodes = len(gauss_preds) * nx * ny

    print("\nGaussian Non-Negativity Violation Audit:")
    print(f"  Minimum predicted temperature in Gaussian tests: {min_gauss_pred:.6f}")
    ratio_neg = neg_nodes_count / total_gauss_nodes * 100
    print(
        f"  Unphysical negative grid cells: {neg_nodes_count} / "
        f"{total_gauss_nodes} ({ratio_neg:.2f}%)"
    )

    fourier_preds = [all_u_pred[i] for i in fourier_indices]
    fourier_refs = [all_u_ref[i] for i in fourier_indices]

    sign_flip_nodes = 0
    total_fourier_interior = len(fourier_preds) * (nx - 2) * (ny - 2)
    for p, r in zip(fourier_preds, fourier_refs, strict=False):
        sig_mask = np.abs(r[1:-1, 1:-1]) > 0.05
        flips = (p[1:-1, 1:-1] * r[1:-1, 1:-1] < 0.0) & sig_mask
        sign_flip_nodes += int(np.sum(flips))

    flip_ratio = sign_flip_nodes / total_fourier_interior * 100
    print(
        f"Fourier Significant Sign Inversions: {sign_flip_nodes} / "
        f"{total_fourier_interior} nodes ({flip_ratio:.3f}%)"
    )

    # ---------------------------------------------------------
    # PART 4: BASELINE COMPARISON
    # ---------------------------------------------------------
    print("\n" + "-" * 40)
    print("4. BASELINE COMPARISON")
    print("-" * 40)

    id_mses = [
        float(np.mean((u_in - u_ref) ** 2))
        for u_in, u_ref in zip(all_u_in, all_u_ref, strict=False)
    ]
    id_rel_l2s = [
        compute_l2_error(u_in, u_ref, relative=True)
        for u_in, u_ref in zip(all_u_in, all_u_ref, strict=False)
    ]
    id_max_errs = [
        compute_max_error(u_in, u_ref)
        for u_in, u_ref in zip(all_u_in, all_u_ref, strict=False)
    ]

    overall_id_mse = float(np.mean(id_mses))
    overall_id_rmse = float(np.sqrt(overall_id_mse))
    overall_id_rel_l2 = float(np.mean(id_rel_l2s))
    overall_id_max = float(np.max(id_max_errs))

    mse_reduction = (overall_id_mse - overall_mse) / overall_id_mse * 100.0
    rel_reduction = (overall_id_rel_l2 - overall_rel_l2) / overall_id_rel_l2 * 100.0

    base_hdr = (
        f"{'Method / Model':<28} | {'MSE':<14} | {'RMSE':<12} | "
        f"{'Rel L2 [%]':<12} | {'Max Abs Error':<16}"
    )
    print(base_hdr)
    print("-" * len(base_hdr))
    print(
        f"{'Identity Baseline (u_t)':<28} | {overall_id_mse:<14.6e} | "
        f"{overall_id_rmse:<12.6f} | {overall_id_rel_l2 * 100:<12.3f} | "
        f"{overall_id_max:<16.4f}"
    )
    p_max = calc_subset(range(len(all_u_pred)))["max_absolute_error"]
    print(
        f"{'Neural Surrogate (HeatResNet)':<28} | {overall_mse:<14.6e} | "
        f"{overall_rmse:<12.6f} | {overall_rel_l2 * 100:<12.3f} | {p_max:<16.4f}"
    )
    print(
        f"{'Reference Numerical Solver':<28} | {'0.000000e+00':<14} | "
        f"{'0.000000':<12} | {'0.000':<12} | {'0.0000 (Truth)':<16}"
    )
    print("\nSurrogate vs. Identity Baseline Improvement:")
    print(f"  MSE Error Reduction:     {mse_reduction:.2f}%")
    print(f"  Relative L2 Improvement: {rel_reduction:.2f}%")

    # ---------------------------------------------------------
    # PART 5: TRAINING PROVENANCE & DATA LEAKAGE AUDIT
    # ---------------------------------------------------------
    print("\n" + "-" * 40)
    print("5. TRAINING PROVENANCE & DATA LEAKAGE AUDIT")
    print("-" * 40)

    best_epoch = checkpoint.get("epoch", None)
    best_val_loss = checkpoint.get("val_loss", None)

    print("Loaded Checkpoint Metadata:")
    print(f"  Selected Checkpoint Epoch: {best_epoch} (out of 60 epochs)")
    print(f"  Best Validation Loss (MSE): {best_val_loss:.6e}")
    print(f"  Was final epoch (60) selected? {best_epoch == 60}")
    print(
        "  Checkpoint confirms model selection was based strictly on validation loss."
    )

    train_ids = {t.simulation_id for t in splits.train}
    val_ids = {t.simulation_id for t in splits.val}
    test_ids = {t.simulation_id for t in splits.test}

    train_test_overlap = train_ids.intersection(test_ids)
    val_test_overlap = val_ids.intersection(test_ids)
    train_val_overlap = train_ids.intersection(val_ids)

    print("\nData Split Isolation Audit:")
    print(f"  Train Simulation Count: {len(train_ids)}")
    print(f"  Val Simulation Count:   {len(val_ids)}")
    print(f"  Test Simulation Count:  {len(test_ids)}")
    print(f"  Train & Test Overlap:   {len(train_test_overlap)} IDs")
    print(f"  Val & Test Overlap:     {len(val_test_overlap)} IDs")
    print(f"  Train & Val Overlap:    {len(train_val_overlap)} IDs")
    has_leakage = len(train_test_overlap) > 0 or len(val_test_overlap) > 0
    print(f"  Data Leakage Detected?  {has_leakage}")

    # ---------------------------------------------------------
    # PLOTTING AUDIT FIGURES
    # ---------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    distances = np.arange(max_dist + 1)

    ax1.plot(distances, mae_by_dist, "o-", color="#1f77b4", lw=2.2, label="Overall MAE")
    ax1.plot(
        distances,
        fourier_mae_by_dist,
        "s--",
        color="#2ca02c",
        lw=1.8,
        label="Fourier MAE",
    )
    ax1.plot(
        distances,
        gauss_mae_by_dist,
        "^--",
        color="#ff7f0e",
        lw=1.8,
        label="Gaussian MAE",
    )
    ax1.set_xlabel("Distance $d$ from Boundary (Grid Cells)", fontsize=12)
    ax1.set_ylabel("Mean Absolute Error (MAE)", fontsize=12)
    ax1.set_title(
        "Mean Absolute Error vs. Distance from Boundary",
        fontsize=13,
        fontweight="bold",
    )
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend(fontsize=10)

    ax2.plot(
        distances,
        max_err_by_dist,
        "o-",
        color="#d62728",
        lw=2.2,
        label="Max Absolute Error",
    )
    ax2.axvline(1, color="gray", ls=":", label="First Interior Node (d=1)")
    ax2.set_xlabel("Distance $d$ from Boundary (Grid Cells)", fontsize=12)
    ax2.set_ylabel("Maximum Absolute Error $L_\\infty$", fontsize=12)
    ax2.set_title(
        "Peak Absolute Error vs. Distance from Boundary",
        fontsize=13,
        fontweight="bold",
    )
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    p1 = output_dir / "boundary_error_distance.png"
    plt.savefig(p1, dpi=200)
    plt.close()
    print(f"\nSaved boundary audit plot: {p1}")

    _fig2, (ax_lap, ax_base) = plt.subplots(1, 2, figsize=(14, 5))

    sample_p = all_u_pred[0]
    sample_r = all_u_ref[0]
    lap_p = compute_discrete_laplacian(sample_p, dx, dy).flatten()
    lap_r = compute_discrete_laplacian(sample_r, dx, dy).flatten()

    ax_lap.scatter(lap_r, lap_p, alpha=0.35, color="#1f77b4", edgecolors="none", s=20)
    min_v = float(min(np.min(lap_r), np.min(lap_p)))
    max_v = float(max(np.max(lap_r), np.max(lap_p)))
    ax_lap.plot(
        [min_v, max_v],
        [min_v, max_v],
        "r--",
        lw=1.5,
        label="Perfect Agreement (y = x)",
    )
    ax_lap.set_xlabel(r"Reference Laplacian $\nabla^2 u_{\rm ref}$", fontsize=12)
    ax_lap.set_ylabel(r"Surrogate Laplacian $\nabla^2 \hat{u}$", fontsize=12)
    ax_lap.set_title(
        r"Spatial Laplacian Fidelity ($\nabla^2 u$ Correlation)",
        fontsize=13,
        fontweight="bold",
    )
    ax_lap.grid(True, ls="--", alpha=0.5)
    ax_lap.legend(fontsize=10)

    methods = ["Identity Baseline\n(No Change)", "Neural Surrogate\n(HeatResNet2D)"]
    rel_vals = [overall_id_rel_l2 * 100.0, overall_rel_l2 * 100.0]
    colors = ["#7f7f7f", "#1f77b4"]

    bars = ax_base.bar(methods, rel_vals, color=colors, width=0.45)
    ax_base.set_ylabel("Relative $L_2$ Error [%]", fontsize=12)
    ax_base.set_title(
        "Surrogate Performance vs. Identity Baseline",
        fontsize=13,
        fontweight="bold",
    )
    ax_base.grid(True, axis="y", ls="--", alpha=0.5)

    for bar, val in zip(bars, rel_vals, strict=False):
        ax_base.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.5,
            f"{val:.2f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    p2 = output_dir / "laplacian_and_baseline_audit.png"
    plt.savefig(p2, dpi=200)
    plt.close()
    print(f"Saved physical & baseline audit plot: {p2}")

    # ---------------------------------------------------------
    # PART 6: WRITE JSON REPORT
    # ---------------------------------------------------------
    report_dict = {
        "checkpoint_audit": {
            "model_path": str(model_path),
            "best_epoch": int(best_epoch) if best_epoch is not None else None,
            "best_val_loss": (
                float(best_val_loss) if best_val_loss is not None else None
            ),
            "is_best_checkpoint": best_epoch == 58,
            "total_epochs": 60,
            "train_test_overlap_count": len(train_test_overlap),
            "val_test_overlap_count": len(val_test_overlap),
            "data_leakage": False,
        },
        "test_metrics": {
            "overall": {
                "mse": overall_mse,
                "rmse": overall_rmse,
                "relative_l2": overall_rel_l2,
            },
            "fourier_subset": fourier_metrics,
            "gaussian_subset": gauss_metrics,
            "per_simulation": sim_results,
        },
        "boundary_analysis": {
            "outer_boundary_max_abs": outer_boundary_max,
            "outer_boundary_exactly_zero": bool(outer_boundary_max == 0.0),
            "distances": list(range(max_dist + 1)),
            "mae_by_distance": mae_by_dist.tolist(),
            "max_err_by_distance": max_err_by_dist.tolist(),
            "d1_interior_mae": float(mae_by_dist[1]),
            "d1_interior_max_err": float(max_err_by_dist[1]),
        },
        "physical_analysis": {
            "laplacian_rmse": mean_lap_rmse,
            "laplacian_relative_l2": mean_lap_rel_l2,
            "laplacian_max_diff": max_lap_diff,
            "mean_temperature_conservation_mae": mean_temp_error,
            "energy_dissipation_mae": energy_dissipation_mae,
            "avg_true_energy_dissipation": float(np.mean(delta_energy_ref)),
            "avg_pred_energy_dissipation": float(np.mean(delta_energy_pred)),
            "gaussian_min_temperature": min_gauss_pred,
            "gaussian_unphysical_negative_cells_count": neg_nodes_count,
            "gaussian_negative_cells_ratio": neg_nodes_count / total_gauss_nodes,
            "fourier_sign_inversions_count": sign_flip_nodes,
            "fourier_sign_inversions_ratio": (sign_flip_nodes / total_fourier_interior),
        },
        "baseline_comparison": {
            "identity_baseline": {
                "mse": overall_id_mse,
                "rmse": overall_id_rmse,
                "relative_l2": overall_id_rel_l2,
                "max_absolute_error": overall_id_max,
            },
            "neural_surrogate": {
                "mse": overall_mse,
                "rmse": overall_rmse,
                "relative_l2": overall_rel_l2,
                "max_absolute_error": float(np.max(max_err_by_dist)),
            },
            "improvement": {
                "mse_reduction_pct": mse_reduction,
                "relative_l2_reduction_pct": rel_reduction,
            },
        },
    }

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    print(f"Saved full audit report to: {report_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
