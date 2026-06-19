"""Regression metrics package."""

from .compute import compute_dummy_regression_preds
from .metrics import build_regression_metrics, mae, r2, rmse
from .plot import (
    get_predicted_vs_actual_scatter_figure,
    get_residual_distribution_figure,
    get_residual_vs_predicted_scatter_figure,
    get_top_k_worst_errors,
)

__all__ = [
    "build_regression_metrics",
    "compute_dummy_regression_preds",
    "get_predicted_vs_actual_scatter_figure",
    "get_residual_vs_predicted_scatter_figure",
    "get_residual_distribution_figure",
    "get_top_k_worst_errors",
    "mae",
    "r2",
    "rmse",
]
