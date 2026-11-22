"""Evaluation metrics for comparing fields, benchmarks, and surrogates."""

import numpy as np


def compute_l2_error(
    u_pred: np.ndarray,
    u_true: np.ndarray,
    relative: bool = False,
) -> float:
    """Compute discrete spatial L2 error norm between predicted and reference fields.

    Args:
        u_pred: Predicted temperature field array.
        u_true: Ground truth or benchmark temperature field array of identical shape.
        relative: If True, normalizes by the L2 norm of u_true.

    Returns:
        Scalar L2 error value.
    """
    diff = u_pred - u_true
    l2_abs = float(np.sqrt(np.mean(diff**2)))
    if relative:
        norm_ref = float(np.sqrt(np.mean(u_true**2)))
        if norm_ref == 0.0:
            raise ValueError(
                "Reference field has zero L2 norm; relative L2 error is undefined."
            )
        return l2_abs / norm_ref
    return l2_abs


def compute_max_error(u_pred: np.ndarray, u_true: np.ndarray) -> float:
    """Compute the discrete L-infinity (maximum point-wise absolute) error.

    Args:
        u_pred: Predicted temperature field array.
        u_true: Ground truth or benchmark temperature field array.

    Returns:
        Maximum absolute point-wise deviation.
    """
    return float(np.max(np.abs(u_pred - u_true)))


def compute_rmse(u_pred: np.ndarray, u_true: np.ndarray) -> float:
    """Compute the Root Mean Square Error (RMSE) across all grid nodes."""
    return float(np.sqrt(np.mean((u_pred - u_true) ** 2)))
