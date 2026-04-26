"""Task-local metrics and error helpers for ordinal score prediction."""

from __future__ import annotations

from functools import partial
from typing import Sequence

import numpy as np
from dlordinal.metrics import amae as amae_score
from dlordinal.metrics import mmae as mmae_score
from sklearn.metrics import cohen_kappa_score, mean_absolute_error

from aitraf.metrics.pipeline import EvalMetric

amae = EvalMetric(name="amae", callback=amae_score)
mae = EvalMetric(name="mae", callback=mean_absolute_error)
mmae = EvalMetric(name="mmae", callback=mmae_score)
qwk = EvalMetric(
    name="qwk",
    callback=partial(cohen_kappa_score, weights="quadratic"),
)


def compute_constant_median_pred_ids(labels: Sequence[int], count: int) -> np.ndarray:
    """Return a constant ordinal baseline using the median class id."""

    label_array = np.asarray(labels, dtype=int)
    median_label = int(np.floor(np.median(label_array)))
    return np.full(count, median_label, dtype=int)


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
    "amae",
    "compute_constant_median_pred_ids",
    "get_top_k_worst_ordinal_errors",
    "mae",
    "mmae",
    "qwk",
]
