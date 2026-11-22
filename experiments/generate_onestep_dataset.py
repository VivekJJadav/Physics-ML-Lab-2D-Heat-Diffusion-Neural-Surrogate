"""CLI script to generate, partition, and save one-step FTCS simulation datasets.

Milestone 5 Requirement:
Saves consecutive FTCS solver states (dt = 0.005s) across 0.5s duration
(101 snapshots, 100 consecutive step pairs per simulation).
Maintains simulation-level 28/6/6 split isolation.
"""

import argparse
from pathlib import Path

from physics_ml.config import SimulationConfig
from physics_ml.data import (
    FourierModeGenerator,
    GaussianSourceGenerator,
    SimulationDataPipeline,
    save_dataset,
)


def main() -> None:
    """CLI entry point for one-step dataset synthesis."""
    parser = argparse.ArgumentParser(
        description="Generate 2D heat diffusion consecutive one-step FTCS dataset."
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
        "--duration",
        type=float,
        default=0.5,
        help="Simulation duration in seconds (default: 0.5s for 100 steps).",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.005,
        help="Internal FTCS solver time step (default: 0.005s).",
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
        default="outputs/datasets/heat_diffusion_onestep_v1",
        help="Destination directory for serialized .npz splits and metadata.json.",
    )

    args = parser.parse_args()

    num_steps = round(args.duration / args.dt)
    num_snapshots = num_steps + 1

    print("=" * 75)
    print("Milestone 5: One-Step Consecutive FTCS Dataset Generation")
    print("=" * 75)
    print(f"Total simulations    : {args.num_simulations}")
    print(
        f"Simulation duration  : {args.duration}s ({num_steps} steps at dt={args.dt}s)"
    )
    print(f"Snapshots per sim    : {num_snapshots} consecutive states")
    print(
        f"Step interval dt     : {args.dt}s (every single FTCS solver update recorded)"
    )
    g_ratio = 1.0 - args.fourier_ratio
    print(f"Fourier / Gaussian   : {args.fourier_ratio:.2f} / {g_ratio:.2f}")
    print(
        f"Split allocation     : Train={args.train_ratio:.2f} (28 sims), "
        f"Val={args.val_ratio:.2f} (6 sims), Test={args.test_ratio:.2f} (6 sims)"
    )
    print(f"Random seed          : {args.seed}")
    print(f"Output directory     : {args.output_dir}")
    print("-" * 75)

    config = SimulationConfig(
        plate_width=1.0,
        plate_height=1.0,
        grid_resolution_x=51,
        grid_resolution_y=51,
        thermal_diffusivity=0.01,
        simulation_duration=args.duration,
        time_step=args.dt,
    )

    pipeline = SimulationDataPipeline(
        config=config,
        fourier_generator=FourierModeGenerator(min_mode=1, max_mode=4),
        gaussian_generator=GaussianSourceGenerator(min_sources=1, max_sources=3),
        num_snapshots=num_snapshots,
    )

    print("Executing FTCS solver to generate consecutive solver states...")
    splits = pipeline.generate_dataset(
        num_simulations=args.num_simulations,
        fourier_ratio=args.fourier_ratio,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    print(
        f"Partitioned trajectories: {len(splits.train)} train, "
        f"{len(splits.val)} val, {len(splits.test)} test."
    )
    total_pairs = (len(splits.train) + len(splits.val) + len(splits.test)) * (
        num_snapshots - 1
    )
    print(f"Total one-step input-target training pairs: {total_pairs} pairs.")

    repo_root = Path(__file__).resolve().parent.parent
    output_path = (
        (repo_root / args.output_dir)
        if not Path(args.output_dir).is_absolute()
        else Path(args.output_dir)
    )
    print(f"Serializing splits to {output_path}...")
    save_dataset(splits, output_path)

    print("One-step dataset generation complete!")


if __name__ == "__main__":
    main()
