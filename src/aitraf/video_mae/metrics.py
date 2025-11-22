"""Metrics utilities for VideoMAE training and evaluation."""

from typing import List

import evaluate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay, f1_score


def build_compute_metrics():
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def _compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        acc = accuracy.compute(
            predictions=preds,
            references=labels,
        )["accuracy"]

        macro_f1 = f1.compute(
            predictions=preds,
            references=labels,
            average="macro",
        )["f1"]

        return {
            "accuracy": acc,
            "f1_macro": macro_f1,
        }

    return _compute_metrics


def get_target_distribution_figure(
    logits: List[float],
    label_ids: List[int],
    label_names: List[str],
    id2label: dict[str, str],
) -> Figure:
    preds = np.argmax(logits, axis=-1)

    actual = [id2label[str(x)] for x in label_ids]
    predicted = [id2label[str(x)] for x in preds]

    actual_counts = pd.Series(actual).value_counts().reindex(label_names, fill_value=0)
    pred_counts = pd.Series(predicted).value_counts().reindex(label_names, fill_value=0)

    df_plot = pd.DataFrame(
        {"actual": actual_counts, "predicted": pred_counts}, index=label_names
    )

    ax = df_plot.plot.bar(rot=0)
    ax.set_title("Predicted vs actual counts")

    return ax.get_figure()


def get_confusion_matrix_figure(
    logits: List[float],
    label_ids: List[int],
    label_names: List[str],
) -> Figure:
    preds = np.argmax(logits, axis=-1)

    fig, ax = plt.subplots(figsize=(6, 5))

    ConfusionMatrixDisplay.from_predictions(
        label_ids,
        preds,
        labels=range(len(label_names)),
        display_labels=label_names,
        ax=ax,
        cmap="Blues",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()

    return fig


def get_per_class_f1_figure(
    logits,
    label_ids,
    label_names,
):
    preds = np.argmax(logits, axis=-1)

    f1_per_class = f1_score(
        label_ids,
        preds,
        average=None,
        labels=range(len(label_names)),
    )

    df = pd.DataFrame({"label": label_names, "f1": f1_per_class}).set_index("label")

    ax = df.plot.bar(rot=45)
    ax.set_title("Per-class F1")
    ax.set_ylabel("F1")
    ax.set_xlabel("Class")
    plt.tight_layout()

    return ax.get_figure()
