"""Ordinal classification metrics utilities."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from dlordinal.metrics import amae, mmae, ranked_probability_score
from sklearn.metrics import cohen_kappa_score, mean_absolute_error


def compute_ordinal_pred_ids(logits: Sequence[Sequence[float]]) -> np.ndarray:
    """Return class ids from ordinal-classification logits."""

    return np.argmax(np.asarray(logits), axis=-1)


def compute_ordinal_probabilities(logits: Sequence[Sequence[float]]) -> np.ndarray:
    """Return softmax probabilities from logits."""

    logits_array = np.asarray(logits, dtype=np.float64)
    logits_array = logits_array - logits_array.max(axis=-1, keepdims=True)
    exp_logits = np.exp(logits_array)
    return exp_logits / exp_logits.sum(axis=-1, keepdims=True)


def build_ordinal_training_metrics():
    """Build the compact metric set used during training."""

    def _compute_metrics(
        pred_ids: Sequence[int],
        labels: Sequence[int],
    ) -> dict[str, float]:
        pred_ids_array = np.asarray(pred_ids, dtype=int)
        labels_array = np.asarray(labels, dtype=int)
        return {
            "amae": float(amae(labels_array, pred_ids_array)),
            "mae": float(mean_absolute_error(labels_array, pred_ids_array)),
        }

    return _compute_metrics


def build_ordinal_eval_metrics():
    """Build the fuller ordinal metric set used during evaluation."""

    def _compute_metrics(
        pred_ids: Sequence[int],
        labels: Sequence[int],
        pred_probs: Sequence[Sequence[float]],
    ) -> dict[str, float]:
        pred_ids_array = np.asarray(pred_ids, dtype=int)
        labels_array = np.asarray(labels, dtype=int)
        pred_probs_array = np.asarray(pred_probs, dtype=np.float64)
        return {
            "amae": float(amae(labels_array, pred_ids_array)),
            "mae": float(mean_absolute_error(labels_array, pred_ids_array)),
            "mmae": float(mmae(labels_array, pred_ids_array)),
            "qwk": float(
                cohen_kappa_score(labels_array, pred_ids_array, weights="quadratic")
            ),
            "ranked_probability_score": float(
                ranked_probability_score(labels_array, pred_probs_array)
            ),
        }

    return _compute_metrics


def compute_constant_median_pred_ids(labels: Sequence[int], count: int) -> np.ndarray:
    """Return a constant ordinal baseline using the median class id."""

    label_array = np.asarray(labels, dtype=int)
    median_label = int(np.floor(np.median(label_array)))
    return np.full(count, median_label, dtype=int)


def compute_prior_probabilities(labels: Sequence[int], num_classes: int) -> np.ndarray:
    """Return the empirical class-prior probability vector."""

    label_array = np.asarray(labels, dtype=int)
    counts = np.bincount(label_array, minlength=num_classes).astype(np.float64)
    return counts / counts.sum()


def get_top_k_worst_ordinal_errors(
    pred_ids: Sequence[int],
    labels: Sequence[int],
    examples_df,
    id2label: dict[str, str],
    top_k: int = 5,
):
    """Return metadata for the largest ordinal misses."""

    pred_ids_array = np.asarray(pred_ids, dtype=int)
    labels_array = np.asarray(labels, dtype=int)

    df = examples_df.copy()
    df["pred_id"] = pred_ids_array
    df["pred_score"] = df["pred_id"].map(lambda idx: id2label[str(idx)])
    df["actual_id"] = labels_array
    df["actual_score"] = df["actual_id"].map(lambda idx: id2label[str(idx)])
    df["ordinal_error"] = df["pred_id"] - df["actual_id"]
    df["abs_ordinal_error"] = df["ordinal_error"].abs()

    return df.sort_values("abs_ordinal_error", ascending=False).head(top_k)


__all__ = [
    "build_ordinal_eval_metrics",
    "build_ordinal_training_metrics",
    "compute_constant_median_pred_ids",
    "compute_ordinal_pred_ids",
    "compute_ordinal_probabilities",
    "compute_prior_probabilities",
    "get_top_k_worst_ordinal_errors",
]
