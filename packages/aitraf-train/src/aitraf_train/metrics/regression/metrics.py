"""Regression metric definitions."""

from typing import Callable, Sequence

from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from ..metrics import EvalMetric

mae = EvalMetric(
    name="mae",
    callback=mean_absolute_error,
)

rmse = EvalMetric(
    name="rmse",
    callback=root_mean_squared_error,
)

r2 = EvalMetric(
    name="r2",
    callback=r2_score,
)


def build_regression_metrics() -> Callable[
    [Sequence[float], Sequence[float]], dict[str, float]
]:
    def _compute_metrics(
        predictions: Sequence[float],
        labels: Sequence[float],
    ) -> dict[str, float]:
        return {
            "mae": mean_absolute_error(labels, predictions),
            "rmse": root_mean_squared_error(labels, predictions),
            "r2": r2_score(labels, predictions),
        }

    return _compute_metrics


__all__ = [
    "build_regression_metrics",
    "mae",
    "r2",
    "rmse",
]
