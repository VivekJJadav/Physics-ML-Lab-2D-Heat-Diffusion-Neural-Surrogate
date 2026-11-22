"""Evaluation module: Verification metrics, benchmarks, and error diagnostics."""

from physics_ml.evaluation.diagnostics import (
    check_energy_monotonicity,
    check_maximum_principle,
    compute_energy_trajectory,
    compute_l2_energy,
    compute_radial_power_spectrum,
    compute_total_energy,
    run_ood_stress_tests,
)
from physics_ml.evaluation.metrics import (
    compute_l2_error,
    compute_max_error,
    compute_rmse,
)
from physics_ml.evaluation.rollout import RolloutEvaluator

__all__ = [
    "RolloutEvaluator",
    "check_energy_monotonicity",
    "check_maximum_principle",
    "compute_energy_trajectory",
    "compute_l2_energy",
    "compute_l2_error",
    "compute_max_error",
    "compute_radial_power_spectrum",
    "compute_rmse",
    "compute_total_energy",
    "run_ood_stress_tests",
]
