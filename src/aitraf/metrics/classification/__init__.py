"""Classification metrics package."""

from .compute import (
    compute_dummy_classification_pred_ids,
    compute_pred_confidences,
    compute_pred_ids,
)
from .metrics import accuracy, balanced_accuracy, f1_binary, f1_macro, precision, recall
from .plot import (
    get_confusion_matrix_figure,
    get_miss_sampling_figure,
    get_miss_sampling_figures,
    get_per_class_f1_figure,
    get_target_distribution_figure,
    get_top_k_worst_misses,
)

__all__ = [
    "accuracy",
    "balanced_accuracy",
    "f1_binary",
    "f1_macro",
    "precision",
    "recall",
    "compute_dummy_classification_pred_ids",
    "compute_pred_confidences",
    "compute_pred_ids",
    "get_confusion_matrix_figure",
    "get_miss_sampling_figure",
    "get_miss_sampling_figures",
    "get_per_class_f1_figure",
    "get_target_distribution_figure",
    "get_top_k_worst_misses",
]
