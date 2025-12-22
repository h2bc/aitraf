"""Regression metrics utilities."""

from typing import Callable, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)

matplotlib.use("Agg")

def compute_dummy_regression_preds(actual_values: Sequence[float]) -> np.ndarray:
    """Return a constant prediction equal to the mean of the labels."""
    values = np.asarray(actual_values, dtype=np.float32)
    return np.full(values.shape, values.mean())


def build_regression_metrics() -> Callable[
    [Sequence[float], Sequence[float]], dict[str, float]
]:
    def _compute_metrics(
        predictions: Sequence[float], labels: Sequence[float]
    ) -> dict[str, float]:
        mae = mean_absolute_error(labels, predictions)
        rmse = root_mean_squared_error(labels, predictions)
        r2 = r2_score(labels, predictions)

        return {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        }

    return _compute_metrics


def get_predicted_vs_actual_scatter_figure(
    predictions: Sequence[float], labels: Sequence[float]
) -> Figure:
    fig, ax = plt.subplots(figsize=(6, 5))

    sns.scatterplot(x=labels, y=predictions, alpha=0.6, ax=ax)
    min_val = float(min(np.min(predictions), np.min(labels)))
    max_val = float(max(np.max(predictions), np.max(labels)))
    
    ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs actual")
    fig.tight_layout()

    return fig


def get_residual_vs_predicted_scatter_figure(
    predictions: Sequence[float], labels: Sequence[float]
) -> Figure:
    fig, ax = plt.subplots(figsize=(6, 5))

    residuals = np.asarray(labels) - np.asarray(predictions)
    sns.scatterplot(x=predictions, y=residuals, alpha=0.6, ax=ax)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (Actual - Predicted)")
    ax.set_title("Residuals vs predicted")
    fig.tight_layout()

    return fig


__all__ = [
    "build_regression_metrics",
    "compute_dummy_regression_preds",
    "get_predicted_vs_actual_scatter_figure",
    "get_residual_vs_predicted_scatter_figure",
]
