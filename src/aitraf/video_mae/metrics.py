"""Metrics utilities for VideoMAE training and evaluation."""

from typing import Callable, List

import evaluate
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


def build_compute_metrics() -> Callable:
    """Return a compute_metrics function used for both training and eval."""
    accuracy_metric = evaluate.load("accuracy")
    weighted_metrics = evaluate.combine(["precision", "recall", "f1"])

    def _compute_metrics(eval_pred):
        logits, label_ids = eval_pred
        preds = np.argmax(logits, axis=-1)

        accuracy = accuracy_metric.compute(
            predictions=preds,
            references=label_ids,
        )["accuracy"]

        weighted = weighted_metrics.compute(
            predictions=preds,
            references=label_ids,
            average="weighted",
        )

        return {
            "accuracy": accuracy,
            "precision_weighted": weighted["precision"],
            "recall_weighted": weighted["recall"],
            "f1_weighted": weighted["f1"],
        }

    return _compute_metrics


def get_target_distribution_figure(
    logits: List[float], label_ids: List[int], id2label: dict[str, str]
) -> Figure:
    preds = np.array(logits).argmax(axis=-1)

    actual = [id2label[str(x)] for x in label_ids]
    predicted = [id2label[str(x)] for x in preds]

    names = [id2label[str(i)] for i in range(len(id2label))]

    actual_counts = pd.Series(actual).value_counts().reindex(names, fill_value=0)
    pred_counts = pd.Series(predicted).value_counts().reindex(names, fill_value=0)

    df_plot = pd.DataFrame(
        {"actual": actual_counts, "predicted": pred_counts}, index=names
    )

    ax = df_plot.plot.bar(rot=0)
    ax.set_title("Predicted vs actual counts")

    return ax.get_figure()
