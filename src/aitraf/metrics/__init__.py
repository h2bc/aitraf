"""Metrics helpers grouped by classification/regression."""

from .classification import (
    build_classification_metrics,
    compute_dummy_classification_pred_ids,
    compute_pred_confidences,
    compute_pred_ids,
    get_confusion_matrix_figure,
    get_per_class_f1_figure,
    get_target_distribution_figure,
    get_top_k_worst_misses,
)
from .regression import (
    build_regression_metrics,
    compute_dummy_regression_preds,
    get_predicted_vs_actual_scatter_figure,
    get_residual_vs_predicted_scatter_figure,
    get_residual_distribution_figure,
    get_top_k_worst_errors,
)
from .pairwise import build_pairwise_metrics

__all__ = [
    "build_classification_metrics",
    "build_pairwise_metrics",
    "build_regression_metrics",
    "compute_dummy_classification_pred_ids",
    "compute_dummy_regression_preds",
    "get_predicted_vs_actual_scatter_figure",
    "get_residual_vs_predicted_scatter_figure",
    "get_residual_distribution_figure",
    "get_top_k_worst_errors",
    "compute_pred_confidences",
    "compute_pred_ids",
    "get_confusion_matrix_figure",
    "get_per_class_f1_figure",
    "get_target_distribution_figure",
    "get_top_k_worst_misses",
]
