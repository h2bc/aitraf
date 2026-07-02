"""MLflow prediction table export helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import mlflow
import pandas as pd

TRAIN_PREDICTIONS_ARTIFACT = "train_predictions.json"
TEST_PREDICTIONS_ARTIFACT = "test_predictions.json"


def build_prediction_rows(
    examples: pd.DataFrame,
    *,
    predictions: Sequence[int],
    labels: Sequence[int],
    id2label: Mapping[str, str],
    confidences: Sequence[float],
) -> pd.DataFrame:
    """Attach model predictions to test metadata rows."""

    rows = examples.copy().reset_index(drop=True)
    rows["pred_id"] = predictions
    rows["label"] = [id2label[str(value)] for value in predictions]
    rows["actual_id"] = labels
    rows["actual_label"] = [id2label[str(value)] for value in labels]
    rows["confidence"] = confidences

    return rows


def log_train_predictions(rows: pd.DataFrame) -> None:
    """Log the full train prediction table."""

    log_prediction_rows(rows, TRAIN_PREDICTIONS_ARTIFACT)


def log_test_predictions(rows: pd.DataFrame) -> None:
    """Log the full test prediction table under the API's fixed artifact name."""

    log_prediction_rows(rows, TEST_PREDICTIONS_ARTIFACT)


def log_prediction_rows(rows: pd.DataFrame, artifact_file: str) -> None:
    """Log prediction rows to MLflow."""

    mlflow.log_table(rows, artifact_file)


__all__ = [
    "TEST_PREDICTIONS_ARTIFACT",
    "TRAIN_PREDICTIONS_ARTIFACT",
    "build_prediction_rows",
    "log_prediction_rows",
    "log_test_predictions",
    "log_train_predictions",
]
