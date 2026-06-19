"""Immutable metrics pipeline and reporting helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

MetricCallback = Callable[[Sequence[Any], Sequence[Any]], float]


@dataclass(frozen=True)
class EvalMetric:
    name: str
    callback: MetricCallback


@dataclass(frozen=True)
class EvalSet:
    name: str
    predictions: tuple[Any, ...]
    labels: tuple[Any, ...]

    def __init__(
        self,
        *,
        name: str,
        predictions: Sequence[Any],
        labels: Sequence[Any],
    ) -> None:
        pred_values = tuple(predictions)
        label_values = tuple(labels)
        if len(pred_values) != len(label_values):
            raise ValueError(
                f"Set '{name}' has length mismatch: "
                f"{len(pred_values)} predictions vs {len(label_values)} labels."
            )

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "predictions", pred_values)
        object.__setattr__(self, "labels", label_values)


@dataclass(frozen=True)
class EvalModel:
    name: str
    sets: tuple[EvalSet, ...]

    def __init__(self, *, name: str, sets: Sequence[EvalSet]) -> None:
        set_values = tuple(sets)
        set_names = [set_item.name for set_item in set_values]
        if len(set_names) != len(set(set_names)):
            raise ValueError(f"Model '{name}' has duplicate set names: {set_names}.")

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "sets", set_values)


@dataclass(frozen=True)
class EvalSetMetrics:
    name: str
    metrics: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class EvalModelMetrics:
    name: str
    sets: tuple[EvalSetMetrics, ...]


@dataclass(frozen=True)
class EvalModelsMetrics:
    models: tuple[EvalModelMetrics, ...]


EvalModels = Sequence[EvalModel]
EvalMetrics = Sequence[EvalMetric]


def calc_metrics(
    predictions: Sequence[Any],
    labels: Sequence[Any],
    eval_metrics: EvalMetrics,
) -> dict[str, float]:
    """Compute all metrics for one prediction/label sequence pair.

    Metric callbacks are called as ``callback(labels, predictions)``.
    """

    prediction_values = tuple(predictions)
    label_values = tuple(labels)
    if len(prediction_values) != len(label_values):
        raise ValueError(
            "Predictions and labels length mismatch: "
            f"{len(prediction_values)} vs {len(label_values)}."
        )

    return {
        metric.name: float(metric.callback(label_values, prediction_values))
        for metric in eval_metrics
    }


def calc_metrics_for_set(
    eval_set: EvalSet,
    eval_metrics: EvalMetrics,
) -> EvalSetMetrics:
    """Compute metrics for one set."""

    values = tuple(
        calc_metrics(
            predictions=eval_set.predictions,
            labels=eval_set.labels,
            eval_metrics=eval_metrics,
        ).items()
    )
    return EvalSetMetrics(name=eval_set.name, metrics=values)


def calc_metrics_for_model(
    eval_model: EvalModel,
    eval_metrics: EvalMetrics,
) -> EvalModelMetrics:
    """Compute metrics for one model across all sets."""

    set_metrics = tuple(
        calc_metrics_for_set(eval_set, eval_metrics) for eval_set in eval_model.sets
    )

    return EvalModelMetrics(name=eval_model.name, sets=set_metrics)


def calc_metrics_for_models(
    eval_models: EvalModels,
    eval_metrics: EvalMetrics,
) -> EvalModelsMetrics:
    """Compute metrics for all models."""

    return EvalModelsMetrics(
        models=tuple(
            calc_metrics_for_model(eval_model, eval_metrics)
            for eval_model in eval_models
        )
    )


def flatten_metrics_report(metrics: EvalModelsMetrics) -> dict[str, float]:
    """Flatten dataclass metrics into MLflow-friendly flat keys."""

    return {
        f"{set_metrics.name}_{model_metrics.name}_{metric_name}": metric_value
        for model_metrics in metrics.models
        for set_metrics in model_metrics.sets
        for metric_name, metric_value in set_metrics.metrics
    }


def metrics_to_df(metrics_report: EvalModelsMetrics) -> pd.DataFrame:
    metric_columns = tuple(
        dict.fromkeys(
            f"{set_metrics.name}_{metric_name}"
            for model_metrics in metrics_report.models
            for set_metrics in model_metrics.sets
            for metric_name, _ in set_metrics.metrics
        )
    )
    columns = ("model", *metric_columns)
    rows = tuple(
        {
            "model": model_metrics.name,
            **{
                f"{set_metrics.name}_{metric_name}": metric_value
                for set_metrics in model_metrics.sets
                for metric_name, metric_value in set_metrics.metrics
            },
        }
        for model_metrics in metrics_report.models
    )
    return pd.DataFrame(rows, columns=columns)


__all__ = [
    "EvalMetric",
    "EvalMetrics",
    "EvalModel",
    "EvalModels",
    "EvalModelMetrics",
    "EvalModelsMetrics",
    "EvalSet",
    "EvalSetMetrics",
    "calc_metrics",
    "calc_metrics_for_model",
    "calc_metrics_for_models",
    "calc_metrics_for_set",
    "flatten_metrics_report",
    "metrics_to_df",
]
