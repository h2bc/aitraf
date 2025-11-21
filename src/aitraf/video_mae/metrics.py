"""Metrics utilities for VideoMAE training and evaluation."""

from typing import Callable

import evaluate
import numpy as np


def build_compute_metrics() -> Callable:
    """Return a compute_metrics function used for both training and eval."""
    accuracy_metric = evaluate.load("accuracy")
    weighted_metrics = evaluate.combine(["precision", "recall", "f1"])

    def _compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        accuracy = accuracy_metric.compute(
            predictions=preds,
            references=labels,
        )["accuracy"]

        weighted = weighted_metrics.compute(
            predictions=preds,
            references=labels,
            average="weighted",
        )

        return {
            "accuracy": accuracy,
            "precision_weighted": weighted["precision"],
            "recall_weighted": weighted["recall"],
            "f1_weighted": weighted["f1"],
        }

    return _compute_metrics
