"""Physics ML Lab: Scientific machine learning for 2D heat diffusion modeling."""

from physics_ml.config import DEFAULT_SIMULATION_CONFIG, SimulationConfig
from physics_ml.data import (
    DatasetSplits,
    FourierModeGenerator,
    GaussianSourceGenerator,
    HeatDiffusionDataset,
    SimulationDataPipeline,
    SimulationTrajectory,
)
from physics_ml.models import (
    ConvBlock2D,
    HeatResNet2D,
    SurrogateTrainer,
    TrainingHistory,
)
from physics_ml.physics import (
    BoundaryCondition,
    ZeroDirichletBoundary,
    analytical_decay_rate,
    single_mode_analytical_solution,
)
from physics_ml.simulation import (
    FTCSSolver2D,
    Grid2D,
    NumericalInstabilityError,
    SimulationResult,
)

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_SIMULATION_CONFIG",
    "BoundaryCondition",
    "ConvBlock2D",
    "DatasetSplits",
    "FTCSSolver2D",
    "FourierModeGenerator",
    "GaussianSourceGenerator",
    "Grid2D",
    "HeatDiffusionDataset",
    "HeatResNet2D",
    "NumericalInstabilityError",
    "SimulationConfig",
    "SimulationDataPipeline",
    "SimulationResult",
    "SimulationTrajectory",
    "SurrogateTrainer",
    "TrainingHistory",
    "ZeroDirichletBoundary",
    "__version__",
    "analytical_decay_rate",
    "single_mode_analytical_solution",
]
