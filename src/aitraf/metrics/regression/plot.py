"""Regression plotting and error-analysis helpers."""

from typing import Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure

matplotlib.use("Agg")


def get_predicted_vs_actual_scatter_figure(
    predictions: Sequence[float],
    labels: Sequence[float],
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
    predictions: Sequence[float],
    labels: Sequence[float],
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


def get_residual_distribution_figure(
    predictions: Sequence[float],
    labels: Sequence[float],
) -> Figure:
    fig, ax = plt.subplots(figsize=(6, 5))

    residuals = np.asarray(labels) - np.asarray(predictions)
    sns.histplot(residuals, kde=True, ax=ax)
    ax.axvline(0, color="red", linestyle="--")
    ax.set_xlabel("Residual (Actual - Predicted)")
    ax.set_ylabel("Count")
    ax.set_title("Residual distribution")
    fig.tight_layout()

    return fig


def get_top_k_worst_errors(
    predictions: Sequence[float],
    labels: Sequence[float],
    examples_df,
    top_k: int = 5,
):
    """Return metadata describing the highest-error predictions."""

    df = examples_df.copy()
    df["prediction"] = predictions
    df["error"] = df["prediction"] - np.asarray(labels)
    df["abs_error"] = df["error"].abs()

    return df.sort_values("abs_error", ascending=False).head(top_k)


__all__ = [
    "get_predicted_vs_actual_scatter_figure",
    "get_residual_vs_predicted_scatter_figure",
    "get_residual_distribution_figure",
    "get_top_k_worst_errors",
]
