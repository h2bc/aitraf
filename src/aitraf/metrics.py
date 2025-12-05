"""Metrics utilities shared across model pipelines."""

from collections import Counter
from typing import List

import evaluate
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay, f1_score

matplotlib.use("Agg")


def compute_pred_ids(logits: List[float]) -> List[int]:
    return np.argmax(logits, axis=-1)


def compute_pred_confidences(logits: List[float]) -> np.ndarray:
    """Return the max softmax probability for each row of logits."""
    tensor_logits = torch.as_tensor(logits)
    probs = torch.softmax(tensor_logits, dim=-1)
    return probs.max(dim=-1).values.numpy()


def compute_dummy_pred_ids(actual_ids: List[int]) -> List[int]:
    most_common = Counter(actual_ids).most_common(1)[0][0]

    return np.full(len(actual_ids), most_common)


def build_compute_metrics() -> dict[str, float]:
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def _compute_metrics(pred_ids: List[int], actual_ids: List[int]):
        acc = accuracy.compute(
            predictions=pred_ids,
            references=actual_ids,
        )["accuracy"]

        macro_f1 = f1.compute(
            predictions=pred_ids,
            references=actual_ids,
            average="macro",
        )["f1"]

        return {
            "accuracy": acc,
            "f1_macro": macro_f1,
        }

    return _compute_metrics


def get_target_distribution_figure(
    pred_ids: List[int],
    actual_ids: List[int],
    label_names: List[str],
    id2label: dict[str, str],
) -> Figure:
    actual_labels = [id2label[str(x)] for x in actual_ids]
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
    actual_ids: List[int],
    label_names: List[str],
) -> Figure:
    fig, ax = plt.subplots(figsize=(6, 5))

    ConfusionMatrixDisplay.from_predictions(
        actual_ids,
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
    actual_ids: List[int],
    label_names: List[str],
):
    f1_per_class = f1_score(
        actual_ids,
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
    actual_ids,
    eval_dataset,
    id2label: dict[str, str],
    top_k: int = 5,
) -> pd.DataFrame:
    """Return metadata describing the highest-confidence misclassifications."""

    df = eval_dataset.to_pandas()
    df["pred_id"] = compute_pred_ids(pred_logits)
    df["pred_confidence"] = compute_pred_confidences(pred_logits)
    df["pred_trick"] = df["pred_id"].map(lambda idx: id2label[str(idx)])
    df["actual_id"] = actual_ids

    misses = df[df["pred_id"] != df["actual_id"]].copy()
    misses = misses.sort_values("pred_confidence", ascending=False).head(top_k)

    return misses
