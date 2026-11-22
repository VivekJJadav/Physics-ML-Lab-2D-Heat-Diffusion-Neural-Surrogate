"""Models module: Neural network surrogate architectures and training modules."""

from physics_ml.models.onestep_surrogate import OneStepConvBlock2D, OneStepHeatResNet2D
from physics_ml.models.surrogate import ConvBlock2D, HeatResNet2D
from physics_ml.models.trainer import (
    SurrogateTrainer,
    TrainingHistory,
    compute_discrete_laplacian_torch,
    compute_laplacian_reference_scale,
)

__all__ = [
    "ConvBlock2D",
    "HeatResNet2D",
    "OneStepConvBlock2D",
    "OneStepHeatResNet2D",
    "SurrogateTrainer",
    "TrainingHistory",
    "compute_discrete_laplacian_torch",
    "compute_laplacian_reference_scale",
]
