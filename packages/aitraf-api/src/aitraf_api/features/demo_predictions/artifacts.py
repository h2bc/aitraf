"""Download and read demo prediction artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow

TEST_PREDICTIONS_ARTIFACT = "test_predictions.json"


class PredictionArtifactError(RuntimeError):
    """Raised when a configured prediction artifact cannot be used."""


@dataclass(frozen=True)
class PredictionArtifactSource:
    task: str
    run_id: str


def download_prediction_rows(source: PredictionArtifactSource) -> list[dict[str, Any]]:
    path = download_prediction_artifact(source)
    return read_prediction_rows(path)


def download_prediction_artifact(source: PredictionArtifactSource) -> Path:
    if not source.run_id:
        raise PredictionArtifactError(f"{source.task} prediction run ID is required")

    try:
        downloaded_path = mlflow.artifacts.download_artifacts(
            run_id=source.run_id,
            artifact_path=TEST_PREDICTIONS_ARTIFACT,
        )
    except Exception as exc:  # noqa: BLE001
        raise PredictionArtifactError(
            f"Failed to download {source.task} prediction artifact "
            f"{TEST_PREDICTIONS_ARTIFACT!r} from run {source.run_id!r}"
        ) from exc

    path = Path(downloaded_path)
    if not path.exists() or not path.is_file():
        raise PredictionArtifactError(
            f"Downloaded {source.task} prediction artifact is missing: {path}"
        )
    return path


def read_prediction_rows(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise PredictionArtifactError(
            f"Failed to read prediction artifact: {path}"
        ) from exc

    columns = [str(column) for column in payload["columns"]]
    return [dict(zip(columns, values, strict=True)) for values in payload["data"]]


__all__ = [
    "PredictionArtifactError",
    "PredictionArtifactSource",
    "TEST_PREDICTIONS_ARTIFACT",
    "download_prediction_artifact",
    "download_prediction_rows",
    "read_prediction_rows",
]
