"""Automated simulation dataset generation pipeline for 2D heat diffusion."""

from typing import Any

import numpy as np

from physics_ml.config import SimulationConfig
from physics_ml.data.dataset import (
    DatasetSplits,
    SimulationTrajectory,
    split_simulations,
)
from physics_ml.data.generators import (
    FourierModeGenerator,
    GaussianSourceGenerator,
    InitialConditionGenerator,
)
from physics_ml.simulation.ftcs import FTCSSolver2D
from physics_ml.simulation.grid import Grid2D


class SimulationDataPipeline:
    """Pipeline orchestrating IC synthesis, numerical solving, and splitting.

    Attributes:
        config: SimulationConfig defining grid geometry, diffusivity, and time stepping.
        fourier_gen: Initial condition generator for Fourier sine mode superpositions.
        gaussian_gen: Initial condition generator for localized Gaussian sources.
        num_snapshots: Number of snapshots to record across simulation duration.
    """

    def __init__(
        self,
        config: SimulationConfig,
        fourier_generator: FourierModeGenerator | None = None,
        gaussian_generator: GaussianSourceGenerator | None = None,
        num_snapshots: int = 11,
    ) -> None:
        if num_snapshots < 2:
            raise ValueError(f"num_snapshots must be >= 2, got {num_snapshots}")

        self.config = config
        self.grid = Grid2D.from_config(config)
        self.fourier_gen = fourier_generator or FourierModeGenerator()
        self.gaussian_gen = gaussian_generator or GaussianSourceGenerator()
        self.num_snapshots = num_snapshots
        self.solver = FTCSSolver2D(config)

    def _determine_record_interval(self) -> int:
        """Compute the step interval needed to capture approximately num_snapshots."""
        interval = max(1, self.config.num_time_steps // (self.num_snapshots - 1))
        return interval

    def run_single_simulation(
        self,
        generator: InitialConditionGenerator,
        simulation_id: str,
        rng: np.random.Generator,
    ) -> SimulationTrajectory:
        """Execute a single simulation from a generated initial condition.

        Args:
            generator: InitialConditionGenerator to synthesize u0.
            simulation_id: Unique string identifier for provenance.
            rng: NumPy random generator.

        Returns:
            SimulationTrajectory holding snapshots, timestamps, and parameters.
        """
        u0, ic_params = generator.generate(self.grid, rng)
        record_interval = self._determine_record_interval()

        # Run numerical integration
        result = self.solver.solve(u0, record_interval=record_interval)

        return SimulationTrajectory(
            simulation_id=simulation_id,
            generator_type=str(ic_params.get("generator", "custom")),
            ic_params=ic_params,
            timestamps=result.times,
            snapshots=result.states,
            config=self.config,
        )

    def generate_dataset(
        self,
        num_simulations: int,
        fourier_ratio: float = 0.5,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> DatasetSplits:
        """Generate simulation ensemble and partition into train/val/test splits.

        Args:
            num_simulations: Total number of independent simulations to run.
            fourier_ratio: Fraction of initial conditions from Fourier modes.
            train_ratio: Fraction of simulations for training split.
            val_ratio: Fraction of simulations for validation split.
            test_ratio: Fraction of simulations for testing split.
            seed: Master random seed for reproducible sampling and splitting.

        Returns:
            DatasetSplits containing partitioned trajectories and provenance metadata.
        """
        if num_simulations < 3:
            raise ValueError(f"num_simulations must be >= 3, got {num_simulations}")
        if not (0.0 <= fourier_ratio <= 1.0):
            raise ValueError(f"fourier_ratio must be in [0, 1], got {fourier_ratio}")

        rng = np.random.default_rng(seed)
        n_fourier = round(num_simulations * fourier_ratio)
        n_gaussian = num_simulations - n_fourier

        trajectories: list[SimulationTrajectory] = []

        # Run Fourier initial condition ensemble
        for i in range(n_fourier):
            sim_id = f"sim_fourier_{i + 1:04d}"
            traj = self.run_single_simulation(self.fourier_gen, sim_id, rng)
            trajectories.append(traj)

        # Run Gaussian initial condition ensemble
        for i in range(n_gaussian):
            sim_id = f"sim_gaussian_{i + 1:04d}"
            traj = self.run_single_simulation(self.gaussian_gen, sim_id, rng)
            trajectories.append(traj)

        # Partition simulations into disjoint splits
        splits = split_simulations(
            trajectories,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )

        pipeline_meta: dict[str, Any] = {
            "pipeline": "SimulationDataPipeline",
            "num_simulations": num_simulations,
            "fourier_ratio": fourier_ratio,
            "n_fourier": n_fourier,
            "n_gaussian": n_gaussian,
            "num_snapshots": self.num_snapshots,
        }
        splits.metadata.update(pipeline_meta)
        return splits
