"""Classification plotting and error-analysis helpers."""

from typing import List

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay, f1_score

from .compute import compute_pred_confidences, compute_pred_ids

matplotlib.use("Agg")


def get_target_distribution_figure(
    pred_ids: List[int],
    labels: List[int],
    label_names: List[str],
    id2label: dict[str, str],
) -> Figure:
    actual_labels = [id2label[str(x)] for x in labels]
    predicted_labels = [id2label[str(x)] for x in pred_ids]

    actual_counts = (
        pd.Series(actual_labels).value_counts().reindex(label_names, fill_value=0)
    )
    pred_counts = (
        pd.Series(predicted_labels).value_counts().reindex(label_names, fill_value=0)
    )

    df_plot = pd.DataFrame(
        {"actual": actual_counts, "predicted": pred_counts}, index=label_names
    )

    ax = df_plot.plot.bar(rot=0)
    ax.set_title("Predicted vs actual counts")

    return ax.get_figure()


def get_confusion_matrix_figure(
    pred_ids: List[int],
    labels: List[int],
    label_names: List[str],
) -> Figure:
    fig, ax = plt.subplots(figsize=(6, 5))

    ConfusionMatrixDisplay.from_predictions(
        labels,
        pred_ids,
        labels=range(len(label_names)),
        display_labels=label_names,
        ax=ax,
        cmap="Blues",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()

    return fig


def get_per_class_f1_figure(
    pred_ids: List[int],
    labels: List[int],
    label_names: List[str],
):
    f1_per_class = f1_score(
        labels,
        pred_ids,
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


def get_top_k_worst_misses(
    pred_logits,
    labels,
    examples_df: pd.DataFrame,
    id2label: dict[str, str],
    top_k: int = 5,
) -> pd.DataFrame:
    """Return metadata describing the highest-confidence misclassifications."""

    df = examples_df.copy()
    df["pred_id"] = compute_pred_ids(pred_logits)
    df["pred_confidence"] = compute_pred_confidences(pred_logits)
    df["pred_trick"] = df["pred_id"].map(lambda idx: id2label[str(idx)])
    df["actual_id"] = labels

    misses = df[df["pred_id"] != df["actual_id"]].copy()
    misses = misses.sort_values("pred_confidence", ascending=False).head(top_k)

    return misses


__all__ = [
    "get_confusion_matrix_figure",
    "get_per_class_f1_figure",
    "get_target_distribution_figure",
    "get_top_k_worst_misses",
]
