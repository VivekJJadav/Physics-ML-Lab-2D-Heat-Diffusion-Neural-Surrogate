"""CLI script to generate, partition, and save heat diffusion simulation datasets."""

import argparse
from pathlib import Path

from physics_ml.config import DEFAULT_SIMULATION_CONFIG
from physics_ml.data import (
    FourierModeGenerator,
    GaussianSourceGenerator,
    SimulationDataPipeline,
    save_dataset,
)


def main() -> None:
    """CLI entry point for dataset synthesis."""
    parser = argparse.ArgumentParser(
        description="Generate 2D heat diffusion dataset with verified numerical solver."
    )
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=40,
        help="Total number of simulations to synthesize (default: 40).",
    )
    parser.add_argument(
        "--fourier-ratio",
        type=float,
        default=0.5,
        help="Proportion of Fourier-mode initial conditions (default: 0.5).",
    )
    parser.add_argument(
        "--num-snapshots",
        type=int,
        default=11,
        help="Number of snapshots per simulation (default: 11).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Fraction of simulations allocated to training (default: 0.70).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Fraction of simulations allocated to validation (default: 0.15).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Fraction of simulations allocated to testing (default: 0.15).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master random seed for sampling and splitting (default: 42).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/datasets/heat_diffusion_v1",
        help="Destination directory for serialized .npz splits and metadata.json.",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Scientific ML Lab: 2D Heat Diffusion Dataset Generation")
    print("=" * 70)
    print(f"Total simulations : {args.num_simulations}")
    print(
        f"Fourier / Gaussian: {args.fourier_ratio:.2f} / {1.0 - args.fourier_ratio:.2f}"
    )
    print(f"Snapshots per sim : {args.num_snapshots}")
    print(
        f"Split ratios      : Train={args.train_ratio:.2f}, "
        f"Val={args.val_ratio:.2f}, Test={args.test_ratio:.2f}"
    )
    print(f"RNG Seed          : {args.seed}")
    print(f"Output directory  : {args.output_dir}")
    print("-" * 70)

    pipeline = SimulationDataPipeline(
        config=DEFAULT_SIMULATION_CONFIG,
        fourier_generator=FourierModeGenerator(min_mode=1, max_mode=4),
        gaussian_generator=GaussianSourceGenerator(min_sources=1, max_sources=3),
        num_snapshots=args.num_snapshots,
    )

    print("Running numerical solver across initial condition ensembles...")
    splits = pipeline.generate_dataset(
        num_simulations=args.num_simulations,
        fourier_ratio=args.fourier_ratio,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    out_path = save_dataset(splits, Path(args.output_dir))
    print(f"Dataset successfully saved to: {out_path.resolve()}")
    print(f"  Train simulations : {len(splits.train)}")
    print(f"  Val simulations   : {len(splits.val)}")
    print(f"  Test simulations  : {len(splits.test)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
