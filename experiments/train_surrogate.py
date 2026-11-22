"""Experiment script to train surrogate and evaluate on held-out test data."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from physics_ml.data import HeatDiffusionDataset, load_dataset
from physics_ml.models import HeatResNet2D, SurrogateTrainer


def plot_training_history(
    history_dict: dict[str, list[float]], output_path: Path
) -> None:
    """Plot and save training and validation loss curves over epochs."""
    epochs = range(1, len(history_dict["train_losses"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(
        epochs, history_dict["train_losses"], label="Train MSE", color="#1f77b4", lw=2
    )
    plt.plot(
        epochs,
        history_dict["val_losses"],
        label="Validation MSE",
        color="#ff7f0e",
        lw=2,
    )
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Mean Squared Error (MSE)", fontsize=12)
    plt.yscale("log")
    plt.title(
        "Neural Surrogate Training & Validation Loss History",
        fontsize=13,
        fontweight="bold",
    )
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_test_comparisons(
    model: HeatResNet2D,
    test_ds: HeatDiffusionDataset,
    output_path: Path,
) -> None:
    """Generate comparison plots of input, reference, prediction, and error."""
    model.eval()

    # Find one Fourier test sample and one Gaussian test sample
    fourier_sample = None
    gaussian_sample = None

    for idx in range(len(test_ds)):
        sample = test_ds[idx]
        sim_id = str(sample["sim_id"])
        if "fourier" in sim_id and fourier_sample is None:
            fourier_sample = sample
        elif "gaussian" in sim_id and gaussian_sample is None:
            gaussian_sample = sample
        if fourier_sample is not None and gaussian_sample is not None:
            break

    samples_to_plot = [
        ("Fourier Harmonic Case", fourier_sample),
        ("Gaussian Heat-Source Case", gaussian_sample),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    with torch.no_grad():
        for row_idx, (title_prefix, sample) in enumerate(samples_to_plot):
            if sample is None:
                continue

            u_in = sample["input"].unsqueeze(0)  # (1, 1, H, W)
            u_true = sample["target"].squeeze().numpy()  # (H, W)
            u_in_np = sample["input"].squeeze().numpy()
            u_pred = model(u_in).squeeze().numpy()  # (H, W)
            abs_err = np.abs(u_pred - u_true)

            vmin = min(float(u_true.min()), float(u_pred.min()))
            vmax = max(float(u_true.max()), float(u_pred.max()))

            # 1. Input field
            im0 = axes[row_idx, 0].imshow(u_in_np, cmap="inferno", origin="lower")
            axes[row_idx, 0].set_title(f"{title_prefix}\nInput $u(t)$", fontsize=10)
            fig.colorbar(im0, ax=axes[row_idx, 0], fraction=0.046, pad=0.04)

            # 2. Ground truth reference
            im1 = axes[row_idx, 1].imshow(
                u_true, cmap="inferno", vmin=vmin, vmax=vmax, origin="lower"
            )
            axes[row_idx, 1].set_title("Numerical Solver\n$u(t+0.1)$", fontsize=10)
            fig.colorbar(im1, ax=axes[row_idx, 1], fraction=0.046, pad=0.04)

            # 3. Neural surrogate prediction
            im2 = axes[row_idx, 2].imshow(
                u_pred, cmap="inferno", vmin=vmin, vmax=vmax, origin="lower"
            )
            axes[row_idx, 2].set_title(
                "Neural Surrogate\n$\\hat{u}(t+0.1)$", fontsize=10
            )
            fig.colorbar(im2, ax=axes[row_idx, 2], fraction=0.046, pad=0.04)

            # 4. Point-wise absolute error
            im3 = axes[row_idx, 3].imshow(abs_err, cmap="magma", origin="lower")
            axes[row_idx, 3].set_title(
                f"Absolute Error $|u - \\hat{{u}}|$\nMax: {abs_err.max():.4e}",
                fontsize=10,
            )
            fig.colorbar(im3, ax=axes[row_idx, 3], fraction=0.046, pad=0.04)

            for col in range(4):
                axes[row_idx, col].set_xticks([])
                axes[row_idx, col].set_yticks([])

    plt.suptitle(
        "Neural Surrogate One-Step Evolution Predictions on Held-Out Test Set",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    """CLI entry point for surrogate training and evaluation."""
    parser = argparse.ArgumentParser(
        description="Train neural surrogate on heat diffusion."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="outputs/datasets/heat_diffusion_v1",
        help="Path to dataset directory containing train.npz, val.npz, test.npz.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
        help="Number of training epochs (default: 60).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (default: 16).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Initial learning rate for AdamW (default: 1e-3).",
    )
    parser.add_argument(
        "--hidden-channels",
        type=int,
        default=32,
        help="Feature channels in residual blocks (default: 32).",
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=3,
        help="Number of residual blocks (default: 3).",
    )
    parser.add_argument(
        "--output-model-dir",
        type=str,
        default="outputs/models",
        help="Directory to save the best model checkpoint.",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default="outputs/figures",
        help="Directory to save evaluation plots.",
    )

    args = parser.parse_args()

    print("=" * 75)
    print("Milestone 4: Training Neural Surrogate for 2D Heat Diffusion")
    print("=" * 75)

    # 1. Load dataset splits
    print(f"Loading dataset from: {args.dataset_dir}")
    splits = load_dataset(args.dataset_dir)
    print(
        f"Loaded {splits.total_simulations} total simulations: "
        f"{len(splits.train)} train, {len(splits.val)} val, {len(splits.test)} test"
    )

    # 2. Build PyTorch step-mode datasets
    train_ds = HeatDiffusionDataset(splits.train, mode="step")
    val_ds = HeatDiffusionDataset(splits.val, mode="step")
    test_ds = HeatDiffusionDataset(splits.test, mode="step")
    print(
        f"Step pairs: {len(train_ds)} train pairs, "
        f"{len(val_ds)} val pairs, {len(test_ds)} test pairs"
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # 3. Instantiate model
    model = HeatResNet2D(
        in_channels=1,
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
        enforce_zero_boundary=True,
    )
    num_params = model.count_parameters()
    print(f"Instantiated HeatResNet2D with {num_params:,} trainable parameters.")

    # 4. Initialize trainer and scheduler
    checkpoint_path = Path(args.output_model_dir) / "best_surrogate.pt"
    trainer = SurrogateTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.learning_rate,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer,
        T_max=args.epochs,
        eta_min=1e-5,
    )

    # 5. Train model
    print(f"\nTraining for {args.epochs} epochs with Cosine Annealing LR...")
    history = trainer.fit(
        epochs=args.epochs,
        checkpoint_path=checkpoint_path,
        scheduler=scheduler,
    )
    print(
        f"Training complete. Best val MSE: {history.best_val_loss:.6e} "
        f"at epoch {history.best_epoch}"
    )

    # 6. Load best validation model
    print(f"\nLoading best checkpoint: {checkpoint_path}")
    trainer.load_checkpoint(checkpoint_path)

    # 7. Evaluate on Validation Set
    val_metrics = trainer.evaluate(val_loader)
    print("\n" + "-" * 40)
    print("Validation Set Metrics (Model Selection):")
    print(f"  MSE         : {val_metrics['mse']:.6e}")
    print(f"  RMSE        : {val_metrics['rmse']:.6e}")
    print(f"  Relative L2 : {val_metrics['relative_l2'] * 100:.3f}%")
    print("-" * 40)

    # 8. Evaluate on Held-Out Test Set (Overall and Subsets)
    overall_test = trainer.evaluate(test_loader)

    # Split test dataset into Fourier vs Gaussian subsets
    test_fourier_trajs = [t for t in splits.test if t.generator_type == "fourier"]
    test_gaussian_trajs = [t for t in splits.test if t.generator_type == "gaussian"]

    test_fourier_loader = DataLoader(
        HeatDiffusionDataset(test_fourier_trajs, mode="step"),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_gaussian_loader = DataLoader(
        HeatDiffusionDataset(test_gaussian_trajs, mode="step"),
        batch_size=args.batch_size,
        shuffle=False,
    )

    fourier_test_metrics = trainer.evaluate(test_fourier_loader)
    gaussian_test_metrics = trainer.evaluate(test_gaussian_loader)

    print("\nHeld-Out Test Set Evaluation:")
    print(f"  Overall Test MSE        : {overall_test['mse']:.6e}")
    print(f"  Overall Test RMSE       : {overall_test['rmse']:.6e}")
    print(f"  Overall Relative L2     : {overall_test['relative_l2'] * 100:.3f}%")
    print(f"\n  Fourier Test Subset (N={len(test_fourier_trajs)}):")
    print(f"    MSE                   : {fourier_test_metrics['mse']:.6e}")
    print(f"    RMSE                  : {fourier_test_metrics['rmse']:.6e}")
    print(
        f"    Relative L2           : {fourier_test_metrics['relative_l2'] * 100:.3f}%"
    )
    print(f"\n  Gaussian Test Subset (N={len(test_gaussian_trajs)}):")
    print(f"    MSE                   : {gaussian_test_metrics['mse']:.6e}")
    print(f"    RMSE                  : {gaussian_test_metrics['rmse']:.6e}")
    print(
        f"    Relative L2           : {gaussian_test_metrics['relative_l2'] * 100:.3f}%"
    )
    print("-" * 75)

    # 9. Save visualizations and metrics summary
    figures_dir = Path(args.figures_dir)
    history_plot_path = figures_dir / "training_history.png"
    comparisons_plot_path = figures_dir / "surrogate_predictions_test.png"

    plot_training_history(history.to_dict(), history_plot_path)
    plot_test_comparisons(trainer.model, test_ds, comparisons_plot_path)
    print(f"Saved training history plot to: {history_plot_path.resolve()}")
    print(f"Saved prediction comparisons plot to: {comparisons_plot_path.resolve()}")

    # Save metrics JSON summary
    metrics_summary = {
        "best_epoch": history.best_epoch,
        "best_val_loss": history.best_val_loss,
        "validation_metrics": val_metrics,
        "overall_test_metrics": overall_test,
        "fourier_test_metrics": fourier_test_metrics,
        "gaussian_test_metrics": gaussian_test_metrics,
        "num_parameters": num_params,
    }
    metrics_json_path = Path(args.output_model_dir) / "evaluation_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"Saved evaluation metrics summary to: {metrics_json_path.resolve()}")
    print("=" * 75)


if __name__ == "__main__":
    main()
