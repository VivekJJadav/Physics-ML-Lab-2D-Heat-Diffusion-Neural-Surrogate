"""Milestone 5: Multi-Step Autoregressive Rollout & Failure Mode Analysis.

Investigates:
1. Multi-step autoregressive error compounding over time (10 steps, 0.1s to 1.0s).
2. Physical invariant tracking: Total thermal energy, L2 Lyapunov energy.
3. Discrete maximum principle adherence (spurious heating / undershoot).
4. Spectral bias analysis using 2D FFT radial power spectrum.
5. Out-of-Distribution (OOD) stress testing across unseen frequencies and gradients.
6. Generation of publication-quality diagnostic plots and structured metrics report.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from physics_ml.data.dataset import load_dataset
from physics_ml.evaluation.diagnostics import (
    check_energy_monotonicity,
    check_maximum_principle,
    compute_energy_trajectory,
    compute_radial_power_spectrum,
    run_ood_stress_tests,
)
from physics_ml.evaluation.rollout import RolloutEvaluator
from physics_ml.models.surrogate import HeatResNet2D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Milestone 5: Autoregressive Rollout & Failure Analysis"
    )

    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="outputs/datasets/heat_diffusion_v1",
        help="Path to serialized dataset directory.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="outputs/models/best_surrogate.pt",
        help="Path to trained PyTorch surrogate checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/figures",
        help="Directory to save diagnostic visualization figures.",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default="outputs/models/rollout_evaluation_report.json",
        help="Path to save the JSON evaluation metrics report.",
    )
    parser.add_argument(
        "--hidden-channels",
        type=int,
        default=32,
        help="Hidden channels in surrogate architecture (default: 32).",
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=3,
        help="Number of residual blocks in surrogate architecture (default: 3).",
    )
    return parser.parse_args()


def plot_rollout_error_growth(
    ensemble_results: dict[str, Any],
    timestamps: list[float],
    output_path: Path,
) -> None:
    """Plot multi-step relative L2 and Max error growth curves."""
    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    times = timestamps

    # 1. Relative L2 Error Growth
    mean_rel = np.array(ensemble_results["mean_relative_l2"]) * 100.0
    std_rel = np.array(ensemble_results["std_relative_l2"]) * 100.0

    fourier_rel = (
        np.array(ensemble_results["fourier_subset"]["mean_relative_l2"]) * 100.0
    )
    gauss_rel = (
        np.array(ensemble_results["gaussian_subset"]["mean_relative_l2"]) * 100.0
    )

    ax1.plot(
        times,
        mean_rel,
        "o-",
        color="#1f77b4",
        lw=2.5,
        label=f"Overall Mean (Final: {mean_rel[-1]:.2f}%)",
    )
    ax1.fill_between(
        times,
        np.maximum(0.0, mean_rel - std_rel),
        mean_rel + std_rel,
        color="#1f77b4",
        alpha=0.2,
        label=r"Mean $\pm$ 1 Std Dev",
    )
    ax1.plot(
        times,
        fourier_rel,
        "s--",
        color="#2ca02c",
        lw=1.8,
        label=f"Fourier Subset (Final: {fourier_rel[-1]:.2f}%)",
    )
    ax1.plot(
        times,
        gauss_rel,
        "^--",
        color="#ff7f0e",
        lw=1.8,
        label=f"Gaussian Subset (Final: {gauss_rel[-1]:.2f}%)",
    )

    ax1.set_xlabel("Physical Time $t$ [s]", fontsize=12)
    ax1.set_ylabel("Relative $L_2$ Error [%]", fontsize=12)
    ax1.set_title(
        "Autoregressive Relative $L_2$ Error Compounding",
        fontsize=13,
        fontweight="bold",
    )
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend(fontsize=10, loc="upper left")

    # 2. Maximum Point-wise Error Growth
    mean_max = np.array(ensemble_results["mean_max_error"])
    fourier_max = np.array(ensemble_results["fourier_subset"]["mean_max_error"])
    gauss_max = np.array(ensemble_results["gaussian_subset"]["mean_max_error"])

    ax2.plot(
        times,
        mean_max,
        "o-",
        color="#d62728",
        lw=2.5,
        label=f"Overall Mean Max Error ({mean_max[-1]:.4f})",
    )
    ax2.plot(
        times,
        fourier_max,
        "s--",
        color="#2ca02c",
        lw=1.8,
        label="Fourier Mean Max Error",
    )
    ax2.plot(
        times,
        gauss_max,
        "^--",
        color="#ff7f0e",
        lw=1.8,
        label="Gaussian Mean Max Error",
    )

    ax2.set_xlabel("Physical Time $t$ [s]", fontsize=12)
    ax2.set_ylabel("Max Point-wise Absolute Error $L_\\infty$", fontsize=12)
    ax2.set_title(
        "Autoregressive $L_\\infty$ Peak Error Compounding",
        fontsize=13,
        fontweight="bold",
    )
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.legend(fontsize=10, loc="upper left")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved: {output_path}")


def plot_energy_dissipation(
    individual_runs: list[dict[str, Any]],
    dx: float,
    dy: float,
    output_path: Path,
) -> None:
    """Plot physical energy dissipation and surrogate drift."""
    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for run in individual_runs:
        gt_states = run["ground_truth"]
        pred_states = run["predicted"]
        times = run["timestamps"]
        sim_id = run["simulation_id"]
        g_type = run["generator_type"]

        gt_e = compute_energy_trajectory(gt_states, dx, dy)
        pred_e = compute_energy_trajectory(pred_states, dx, dy)

        color = "#1f77b4" if g_type == "fourier" else "#ff7f0e"

        # Panel 1: L2 Lyapunov functional dissipation
        ax1.plot(
            times,
            gt_e["l2_energy"],
            "-",
            color="black",
            alpha=0.3,
            label="FTCS Ground Truth"
            if sim_id == individual_runs[0]["simulation_id"]
            else "",
        )
        ax1.plot(
            times,
            pred_e["l2_energy"],
            "--",
            color=color,
            alpha=0.7,
            label=f"Surrogate ({g_type})"
            if sim_id
            in [
                individual_runs[0]["simulation_id"],
                individual_runs[-1]["simulation_id"],
            ]
            else "",
        )

        # Panel 2: Relative Total Energy Drift
        e0 = gt_e["total_energy"][0]
        drift_pct = (pred_e["total_energy"] - gt_e["total_energy"]) / e0 * 100.0
        ax2.plot(times, drift_pct, "o-", color=color, alpha=0.7, lw=1.5)

    ax1.set_xlabel("Physical Time $t$ [s]", fontsize=12)
    ax1.set_ylabel(r"$L_2$ Energy $\mathcal{H}(t) = \int u^2 d\Omega$", fontsize=12)
    ax1.set_title(
        r"Physical Energy Dissipation: FTCS vs Neural Surrogate",
        fontsize=13,
        fontweight="bold",
    )
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend(fontsize=10)

    ax2.axhline(0.0, color="black", ls=":", lw=1.5)
    ax2.set_xlabel("Physical Time $t$ [s]", fontsize=12)
    ax2.set_ylabel(
        "Total Energy Drift $\\Delta \\mathcal{E} / \\mathcal{E}_0$ [%]", fontsize=12
    )
    ax2.set_title(
        "Thermal Energy Conservation Drift vs Time",
        fontsize=13,
        fontweight="bold",
    )
    ax2.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved: {output_path}")


def plot_failure_spatial_modes(
    worst_run: dict[str, Any],
    output_path: Path,
) -> None:
    """Plot spatial field snapshots and absolute error maps for the worst rollout."""
    gt = worst_run["ground_truth"]
    pred = worst_run["predicted"]
    times = worst_run["timestamps"]
    sim_id = worst_run["simulation_id"]
    gen_type = worst_run["generator_type"]

    # Select snapshot indices: t = 0.1s (step 1), t = 0.5s (step 5), t = 1.0s (step 10)
    step_indices = [1, 5, 10]
    fig, axes = plt.subplots(3, 3, figsize=(13, 11))

    vmax = max(float(np.max(gt[1])), float(np.max(pred[1])))
    vmin = min(float(np.min(gt[1])), float(np.min(pred[1])))

    for col_idx, s_idx in enumerate(step_indices):
        t_val = times[s_idx]
        u_true = gt[s_idx]
        u_pred = pred[s_idx]
        err = np.abs(u_true - u_pred)

        # Row 0: Ground Truth
        im0 = axes[0, col_idx].imshow(
            u_true,
            cmap="inferno",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
        )
        axes[0, col_idx].set_title(
            f"FTCS Truth: $t={t_val:.1f}$s", fontsize=11, fontweight="bold"
        )
        axes[0, col_idx].set_xticks([])
        axes[0, col_idx].set_yticks([])

        # Row 1: Surrogate Prediction
        im1 = axes[1, col_idx].imshow(
            u_pred,
            cmap="inferno",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
        )
        axes[1, col_idx].set_title(
            f"Surrogate: $t={t_val:.1f}$s", fontsize=11, fontweight="bold"
        )
        axes[1, col_idx].set_xticks([])
        axes[1, col_idx].set_yticks([])

        # Row 2: Absolute Error Map
        im2 = axes[2, col_idx].imshow(
            err,
            cmap="magma",
            origin="lower",
        )
        max_e = np.max(err)
        rel_e = worst_run["relative_l2_errors"][s_idx] * 100.0
        axes[2, col_idx].set_title(
            f"Error (Rel: {rel_e:.1f}%, Max: {max_e:.3f})",
            fontsize=10,
            color="#b2182b",
            fontweight="bold",
        )
        axes[2, col_idx].set_xticks([])
        axes[2, col_idx].set_yticks([])
        fig.colorbar(im2, ax=axes[2, col_idx], fraction=0.046, pad=0.04)

    fig.colorbar(im0, ax=axes[0, :], fraction=0.015, pad=0.02)
    fig.colorbar(im1, ax=axes[1, :], fraction=0.015, pad=0.02)

    header = (
        f"Spatial Failure Mode Analysis: Trajectory '{sim_id}' ({gen_type.title()})\n"
        "10-Step Rollout Error Accumulation Across Gradients"
    )

    fig.suptitle(
        header,
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved: {output_path}")


def plot_spectral_bias_and_ood(
    test_runs: list[dict[str, Any]],
    ood_results: dict[str, Any],
    output_path: Path,
) -> None:
    """Plot spectral power distribution and OOD stress test error curves."""
    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Radial Power Spectrum (Ground Truth vs Surrogate at t=1.0s)
    # Average across all test simulations
    k_radial = None
    gt_spectra = []
    pred_spectra = []

    for run in test_runs:
        k_rad, p_gt = compute_radial_power_spectrum(run["ground_truth"][-1])
        _, p_pred = compute_radial_power_spectrum(run["predicted"][-1])
        k_radial = k_rad
        gt_spectra.append(p_gt)
        pred_spectra.append(p_pred)

    mean_p_gt = np.mean(gt_spectra, axis=0)
    mean_p_pred = np.mean(pred_spectra, axis=0)

    if k_radial is not None:
        ax1.semilogy(
            k_radial,
            mean_p_gt,
            "o-",
            color="black",
            lw=2,
            label="FTCS Ground Truth (t=1.0s)",
        )
        ax1.semilogy(
            k_radial,
            mean_p_pred,
            "s--",
            color="#d62728",
            lw=2,
            label="Surrogate Rollout (t=1.0s)",
        )
    ax1.set_xlabel("Radial Spatial Frequency $k_r$", fontsize=12)
    ax1.set_ylabel(r"Averaged Power Density $E(k_r)$", fontsize=12)
    ax1.set_title(
        "Radial Power Spectrum: Spectral Bias Analysis",
        fontsize=13,
        fontweight="bold",
    )
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend(fontsize=10)

    # Panel 2: OOD Stress Testing Error Trajectories
    times = [0.1 * k for k in range(11)]
    for _key, data in ood_results.items():
        rel_pct = np.array(data["rel_l2_history"]) * 100.0
        label = f"{data['description']} (Final: {data['final_rel_l2'] * 100:.1f}%)"
        ax2.plot(times, rel_pct, "o-", lw=2, label=label)

    ax2.set_xlabel("Physical Time $t$ [s]", fontsize=12)
    ax2.set_ylabel("Relative $L_2$ Error [%]", fontsize=12)
    ax2.set_title(
        "Out-of-Distribution (OOD) Rollout Stress Testing",
        fontsize=13,
        fontweight="bold",
    )
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.legend(fontsize=9, loc="upper left")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved: {output_path}")


def main() -> None:
    args = parse_args()
    print("=" * 80)
    print(
        "Milestone 5: Multi-Step Autoregressive Rollout & Physical Invariant Analysis"
    )
    print("=" * 80)

    # 1. Load test dataset splits
    splits = load_dataset(args.dataset_dir)
    test_trajectories = splits.test
    print(f"Loaded test dataset: {len(test_trajectories)} independent trajectories")
    config = test_trajectories[0].config
    dx = config.dx
    dy = config.dy

    # 2. Instantiate and load trained surrogate model
    model = HeatResNet2D(
        in_channels=1,
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
    )
    checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded trained surrogate checkpoint from: {args.model_path}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 3. Perform autonomous multi-step rollouts
    evaluator = RolloutEvaluator(model, device="cpu")
    ensemble_results = evaluator.evaluate_ensemble(test_trajectories)

    print("\n--- Rollout Quantitative Summary (10 Autonomous Steps) ---")
    timestamps = ensemble_results["timestamps"]
    mean_rel = ensemble_results["mean_relative_l2"]
    mean_max = ensemble_results["mean_max_error"]

    print(f"{'Step':<5} | {'Time':<8} | {'Mean Rel L2 [%]':<15} | {'Mean Max Err':<12}")

    print("-" * 50)
    for k in range(len(timestamps)):
        r_l2 = mean_rel[k] * 100
        m_err = mean_max[k]
        print(f"{k:<5} | {timestamps[k]:<8.2f} | {r_l2:<15.3f} | {m_err:<12.4f}")

    fourier_final = ensemble_results["fourier_subset"]["final_mean_relative_l2"] * 100.0
    gauss_final = ensemble_results["gaussian_subset"]["final_mean_relative_l2"] * 100.0
    print("\nFinal Rollout Error by Subset:")
    print(f"  Fourier (4 trajectories):  {fourier_final:.3f}%")
    print(f"  Gaussian (2 trajectories): {gauss_final:.3f}%")

    # 4. Check physical invariants across individual runs
    individual_runs = ensemble_results["individual_runs"]
    energy_diagnostics = []
    max_principle_diagnostics = []

    for run in individual_runs:
        pred_states = run["predicted"]

        pred_energy = compute_energy_trajectory(pred_states, dx, dy)
        mono_check = check_energy_monotonicity(pred_energy["l2_energy"])
        max_p_check = check_maximum_principle(pred_states)

        energy_diagnostics.append(
            {
                "simulation_id": run["simulation_id"],
                "generator_type": run["generator_type"],
                "is_l2_energy_monotonic": mono_check["is_monotonic"],
                "max_energy_violation": mono_check["max_step_increase"],
                "total_energy_drift_pct": float(
                    (pred_energy["total_energy"][-1] - pred_energy["total_energy"][0])
                    / pred_energy["total_energy"][0]
                    * 100.0
                ),
            }
        )

        max_principle_diagnostics.append(
            {
                "simulation_id": run["simulation_id"],
                "generator_type": run["generator_type"],
                "is_adherent": max_p_check["is_adherent"],
                "global_max_violation": max_p_check["global_max_violation"],
                "global_min_violation": max_p_check["global_min_violation"],
                "spurious_heating_steps": max_p_check["spurious_heating_steps"],
            }
        )

    # 5. Out-of-Distribution (OOD) Stress Tests
    print("\n--- Running Out-of-Distribution (OOD) Stress Tests ---")
    ood_results = run_ood_stress_tests(model, config, seed=42)
    for _scn_key, scn_data in ood_results.items():
        print(
            f"  {scn_data['description']}: "
            f"1-step: {scn_data['1step_rel_l2'] * 100:.2f}%, "
            f"10-step: {scn_data['final_rel_l2'] * 100:.2f}%"
        )

    # 6. Generate Figures
    figures_dir = Path(args.output_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_rollout_error_growth(
        ensemble_results,
        timestamps,
        figures_dir / "rollout_error_growth.png",
    )
    plot_energy_dissipation(
        individual_runs,
        dx,
        dy,
        figures_dir / "energy_dissipation_drift.png",
    )

    # Identify worst run for spatial failure mode plot
    worst_run = max(individual_runs, key=lambda r: float(r["final_relative_l2"]))
    plot_failure_spatial_modes(
        worst_run,
        figures_dir / "failure_mode_analysis.png",
    )
    plot_spectral_bias_and_ood(
        individual_runs,
        ood_results,
        figures_dir / "spectral_bias_and_ood.png",
    )

    # 7. Compile and save JSON report
    report_data = {
        "summary": {
            "num_test_simulations": len(test_trajectories),
            "num_steps": len(timestamps) - 1,
            "simulation_duration": config.simulation_duration,
            "1step_mean_relative_l2": float(mean_rel[1]),
            "final_mean_relative_l2": float(mean_rel[-1]),
            "1step_mean_max_error": float(mean_max[1]),
            "final_mean_max_error": float(mean_max[-1]),
            "error_amplification_ratio": float(mean_rel[-1] / max(1e-12, mean_rel[1])),
        },
        "subsets": {
            "fourier": ensemble_results["fourier_subset"],
            "gaussian": ensemble_results["gaussian_subset"],
        },
        "timestamps": timestamps,
        "mean_relative_l2": mean_rel,
        "mean_max_error": mean_max,
        "energy_diagnostics": energy_diagnostics,
        "max_principle_diagnostics": max_principle_diagnostics,
        "ood_stress_tests": {
            k: {
                "description": v["description"],
                "1step_rel_l2": v["1step_rel_l2"],
                "final_rel_l2": v["final_rel_l2"],
                "final_max_error": v["final_max_error"],
                "max_principle_adherent": v["max_principle"]["is_adherent"],
            }
            for k, v in ood_results.items()
        },
    }

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"\nSaved structured evaluation report to: {report_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
