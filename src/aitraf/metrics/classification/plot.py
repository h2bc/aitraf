"""Classification plotting and error-analysis helpers."""

from pathlib import Path
from typing import Iterator, List

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay, f1_score

from aitraf.processing.video import load_sampled_video_frames
from aitraf.utils.s3_utils import build_s3_client, load_s3_settings, presign_s3_uri
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
    top_k: int | None = None,
) -> pd.DataFrame:
    """Return metadata describing misclassifications, optionally capped at top_k."""

    df = examples_df.copy()
    df["pred_id"] = compute_pred_ids(pred_logits)
    df["pred_confidence"] = compute_pred_confidences(pred_logits)
    df["pred_trick"] = df["pred_id"].map(lambda idx: id2label[str(idx)])
    df["actual_id"] = labels

    misses = df[df["pred_id"] != df["actual_id"]].copy()
    misses = misses.sort_values("pred_confidence", ascending=False)

    if top_k is not None:
        misses = misses.head(top_k)

    if "s3_path" in misses.columns and not misses.empty:
        s3_client = build_s3_client(load_s3_settings(require_bucket=False))
        misses["presigned_url"] = misses["s3_path"].map(
            lambda uri: presign_s3_uri(str(uri), s3_client=s3_client)
        )

    return misses


def get_miss_sampling_figures(
    misses: pd.DataFrame,
    *,
    clips_dir: Path | str,
    num_frames: int,
    sampling_dist: str,
    max_cols: int = 4,
) -> Iterator[tuple[str, Figure]]:
    """Yield one sampled-frame contact sheet figure per miss."""

    for _, miss in misses.iterrows():
        frames, frame_indices = load_sampled_video_frames(
            video_id=str(miss["video_id"]),
            local_clips_dir=clips_dir,
            num_frames=num_frames,
            sampling_dist=sampling_dist,
        )
        miss = miss.copy()
        miss["frame_indices"] = frame_indices

        artifact_file = f"misses/{Path(str(miss['video_id'])).stem}/sampling.png"
        figure = get_miss_sampling_figure(
            miss,
            frames,
            max_cols=max_cols,
        )
        try:
            yield artifact_file, figure
        finally:
            plt.close(figure)


def get_miss_sampling_figure(
    miss: pd.Series,
    frames: list,
    *,
    max_cols: int,
) -> Figure:
    cols = min(max_cols, len(frames))
    rows = (len(frames) + cols - 1) // cols
    figure, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * 3.0, rows * 2.4),
        squeeze=False,
    )

    for ax in axes.ravel():
        ax.axis("off")

    for ax, frame, frame_idx in zip(axes.ravel(), frames, miss["frame_indices"]):
        ax.imshow(frame.cpu().numpy())
        ax.set_title(f"frame {frame_idx}", fontsize=10)

    figure.suptitle(
        f"{miss['video_id']} | actual_id={miss['actual_id']} | "
        f"pred_id={miss['pred_id']}",
        fontsize=12,
        y=1.02,
    )
    figure.tight_layout()
    return figure


__all__ = [
    "get_confusion_matrix_figure",
    "get_per_class_f1_figure",
    "get_miss_sampling_figure",
    "get_miss_sampling_figures",
    "get_target_distribution_figure",
    "get_top_k_worst_misses",
]
