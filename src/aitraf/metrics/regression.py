"""Regression metrics utilities."""

from typing import Callable, Sequence

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)


def compute_dummy_regression_preds(actual_values: Sequence[float]) -> np.ndarray:
    """Return a constant prediction equal to the mean of the targets."""
    values = np.asarray(actual_values, dtype=np.float32)
    return np.full(values.shape, values.mean())


def build_regression_metrics() -> Callable[
    [Sequence[float], Sequence[float]], dict[str, float]
]:
    def _compute_metrics(
        predictions: Sequence[float], targets: Sequence[float]
    ) -> dict[str, float]:
        mae = mean_absolute_error(targets, predictions)
        rmse = root_mean_squared_error(targets, predictions)
        r2 = r2_score(targets, predictions)

        return {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        }

    return _compute_metrics


__all__ = [
    "build_regression_metrics",
    "compute_dummy_regression_preds",
]
