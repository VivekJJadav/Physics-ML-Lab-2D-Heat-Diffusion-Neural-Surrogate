"""Dataset representation, splitting, and serialization for 2D heat simulations."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from physics_ml.config import SimulationConfig


@dataclass(frozen=True)
class SimulationTrajectory:
    """Encapsulates a single simulation run and its complete provenance.

    Attributes:
        simulation_id: Unique string identifier (e.g. 'sim_0001').
        generator_type: Initial condition generator ('fourier' or 'gaussian').
        ic_params: Metadata and parameters used to synthesize u0.
        timestamps: 1D array of physical times [s] for recorded snapshots.
        snapshots: 3D array of temperature fields of shape (num_snapshots, Nx, Ny).
        config: SimulationConfig used for numerical integration.
    """

    simulation_id: str
    generator_type: str
    ic_params: dict[str, Any]
    timestamps: np.ndarray
    snapshots: np.ndarray
    config: SimulationConfig

    @property
    def num_snapshots(self) -> int:
        """Number of recorded temporal snapshots."""
        return int(self.snapshots.shape[0])

    @property
    def initial_state(self) -> np.ndarray:
        """Initial temperature field at t = 0 of shape (Nx, Ny)."""
        return self.snapshots[0]

    @property
    def final_state(self) -> np.ndarray:
        """Final temperature field of shape (Nx, Ny)."""
        return self.snapshots[-1]

    def to_metadata_dict(self) -> dict[str, Any]:
        """Serialize simulation provenance to a JSON-compatible dictionary."""
        return {
            "simulation_id": self.simulation_id,
            "generator_type": self.generator_type,
            "ic_params": self.ic_params,
            "timestamps": self.timestamps.tolist(),
            "config": asdict(self.config),
            "grid_shape": list(self.snapshots.shape[1:]),
            "num_snapshots": self.num_snapshots,
        }


@dataclass
class DatasetSplits:
    """Holds train, val, and test trajectory collections partitioned by simulation.

    Attributes:
        train: List of trajectories for model training.
        val: List of trajectories for hyperparameter validation.
        test: List of trajectories held out for final benchmark evaluation.
        metadata: Global dataset provenance, seed, and split statistics.
    """

    train: list[SimulationTrajectory]
    val: list[SimulationTrajectory]
    test: list[SimulationTrajectory]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        """Verify strict simulation isolation across splits."""
        train_ids = set(self.train_ids)
        val_ids = set(self.val_ids)
        test_ids = set(self.test_ids)

        if train_ids & val_ids:
            raise ValueError(
                f"Train and Val splits share simulation IDs: {train_ids & val_ids}"
            )
        if train_ids & test_ids:
            raise ValueError(
                f"Train and Test splits share simulation IDs: {train_ids & test_ids}"
            )
        if val_ids & test_ids:
            raise ValueError(
                f"Val and Test splits share simulation IDs: {val_ids & test_ids}"
            )

    @property
    def total_simulations(self) -> int:
        """Total number of distinct simulation runs across all splits."""
        return len(self.train) + len(self.val) + len(self.test)

    @property
    def train_ids(self) -> list[str]:
        """List of simulation IDs in the training split."""
        return [t.simulation_id for t in self.train]

    @property
    def val_ids(self) -> list[str]:
        """List of simulation IDs in the validation split."""
        return [t.simulation_id for t in self.val]

    @property
    def test_ids(self) -> list[str]:
        """List of simulation IDs in the test split."""
        return [t.simulation_id for t in self.test]


def split_simulations(
    trajectories: list[SimulationTrajectory],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> DatasetSplits:
    """Partition simulations into train, validation, and test splits.

    Splitting is performed strictly at the simulation level (not per-snapshot)
    to prevent temporal data leakage into evaluation splits.

    Args:
        trajectories: Collection of simulation trajectories.
        train_ratio: Target fraction of simulations for training (default: 0.70).
        val_ratio: Target fraction of simulations for validation (default: 0.15).
        test_ratio: Target fraction of simulations for testing (default: 0.15).
        seed: Random seed for deterministic simulation ordering.

    Returns:
        DatasetSplits with non-overlapping simulation partitions.
    """
    total = len(trajectories)
    if total < 3:
        raise ValueError(f"At least 3 simulations are required to split, got {total}")

    ratio_sum = train_ratio + val_ratio + test_ratio
    if not np.isclose(ratio_sum, 1.0, atol=1e-3):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum:.4f}")

    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(total).tolist()

    n_train = round(total * train_ratio)
    n_val = round(total * val_ratio)

    # Ensure all splits contain at least 1 simulation when total >= 3
    n_train = max(1, min(n_train, total - 2))
    n_val = max(1, min(n_val, total - n_train - 1))
    n_test = total - n_train - n_val

    train_trajs = [trajectories[i] for i in shuffled_indices[:n_train]]
    val_trajs = [trajectories[i] for i in shuffled_indices[n_train : n_train + n_val]]
    test_trajs = [trajectories[i] for i in shuffled_indices[n_train + n_val :]]

    metadata = {
        "seed": seed,
        "total_simulations": total,
        "split_counts": {"train": n_train, "val": n_val, "test": n_test},
        "split_fractions": {
            "train": n_train / total,
            "val": n_val / total,
            "test": n_test / total,
        },
    }
    return DatasetSplits(
        train=train_trajs,
        val=val_trajs,
        test=test_trajs,
        metadata=metadata,
    )


def save_dataset(splits: DatasetSplits, dataset_dir: Path | str) -> Path:
    """Save dataset splits to disk as compressed NumPy archives and JSON metadata.

    Args:
        splits: DatasetSplits instance.
        dataset_dir: Destination directory path.

    Returns:
        Path to the saved dataset directory.
    """
    path = Path(dataset_dir)
    path.mkdir(parents=True, exist_ok=True)

    def _pack_split(trajs: list[SimulationTrajectory], filepath: Path) -> None:
        if not trajs:
            return
        ref_times = trajs[0].timestamps
        ref_shape = trajs[0].snapshots.shape[1:]
        ref_cfg = trajs[0].config
        for t in trajs[1:]:
            if len(t.timestamps) != len(ref_times) or not np.allclose(
                t.timestamps, ref_times, atol=1e-12
            ):
                raise ValueError(f"Timestamp mismatch in trajectory {t.simulation_id}.")
            if t.snapshots.shape[1:] != ref_shape:
                raise ValueError(
                    f"Grid shape mismatch in trajectory {t.simulation_id}: "
                    f"{t.snapshots.shape[1:]} vs {ref_shape}."
                )
            if (
                t.config.plate_width != ref_cfg.plate_width
                or t.config.plate_height != ref_cfg.plate_height
                or t.config.thermal_diffusivity != ref_cfg.thermal_diffusivity
            ):
                raise ValueError(
                    f"Configuration parameter mismatch in trajectory {t.simulation_id}."
                )

        snapshots = np.stack([t.snapshots for t in trajs], axis=0)  # (N, K, Nx, Ny)
        times = ref_times
        sim_ids = np.array([t.simulation_id for t in trajs])
        np.savez_compressed(filepath, snapshots=snapshots, times=times, sim_ids=sim_ids)

    _pack_split(splits.train, path / "train.npz")
    _pack_split(splits.val, path / "val.npz")
    _pack_split(splits.test, path / "test.npz")

    # Save complete metadata and per-simulation parameters
    sim_metadata: dict[str, Any] = {}
    for t in splits.train + splits.val + splits.test:
        sim_metadata[t.simulation_id] = t.to_metadata_dict()

    global_meta = {
        **splits.metadata,
        "simulations": sim_metadata,
    }
    with open(path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(global_meta, f, indent=2)

    return path


def load_dataset(dataset_dir: Path | str) -> DatasetSplits:
    """Load dataset splits from disk directory.

    Args:
        dataset_dir: Path to directory containing .npz files and metadata.json.

    Returns:
        DatasetSplits instance populated with simulation trajectories.
    """
    path = Path(dataset_dir)
    with open(path / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)

    sims_meta = metadata.get("simulations", {})

    def _unpack_split(filepath: Path) -> list[SimulationTrajectory]:
        if not filepath.exists():
            return []
        data = np.load(filepath, allow_pickle=False)
        snapshots = data["snapshots"]  # (N, K, Nx, Ny)
        times = data["times"]
        sim_ids = data["sim_ids"]

        trajs: list[SimulationTrajectory] = []
        for i, sid in enumerate(sim_ids):
            sid_str = str(sid)
            meta = sims_meta.get(sid_str, {})
            cfg_dict = meta.get("config", {})
            config = SimulationConfig(
                plate_width=cfg_dict.get("plate_width", 1.0),
                plate_height=cfg_dict.get("plate_height", 1.0),
                grid_resolution_x=cfg_dict.get("grid_resolution_x", 51),
                grid_resolution_y=cfg_dict.get("grid_resolution_y", 51),
                thermal_diffusivity=cfg_dict.get("thermal_diffusivity", 0.01),
                simulation_duration=cfg_dict.get("simulation_duration", 1.0),
                time_step=cfg_dict.get("time_step", 0.005),
            )
            traj = SimulationTrajectory(
                simulation_id=sid_str,
                generator_type=meta.get("generator_type", "unknown"),
                ic_params=meta.get("ic_params", {}),
                timestamps=times,
                snapshots=snapshots[i],
                config=config,
            )
            trajs.append(traj)
        return trajs

    train = _unpack_split(path / "train.npz")
    val = _unpack_split(path / "val.npz")
    test = _unpack_split(path / "test.npz")

    return DatasetSplits(
        train=train,
        val=val,
        test=test,
        metadata=metadata,
    )
