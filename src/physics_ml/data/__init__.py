"""Data module: Initial condition synthesis, pipelines, and PyTorch datasets."""

from physics_ml.data.dataset import (
    DatasetSplits,
    SimulationTrajectory,
    load_dataset,
    save_dataset,
    split_simulations,
)
from physics_ml.data.generators import (
    FourierModeGenerator,
    GaussianSourceGenerator,
    InitialConditionGenerator,
)
from physics_ml.data.pipeline import SimulationDataPipeline
from physics_ml.data.torch_dataset import HeatDiffusionDataset

__all__ = [
    "DatasetSplits",
    "FourierModeGenerator",
    "GaussianSourceGenerator",
    "HeatDiffusionDataset",
    "InitialConditionGenerator",
    "SimulationDataPipeline",
    "SimulationTrajectory",
    "load_dataset",
    "save_dataset",
    "split_simulations",
]
