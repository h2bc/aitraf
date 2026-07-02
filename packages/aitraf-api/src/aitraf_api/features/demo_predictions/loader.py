"""Load and build cached demo prediction responses."""

from __future__ import annotations

from typing import TypedDict

from aitraf_api.config import Settings
from aitraf_api.features.demo_predictions.artifacts import (
    PredictionArtifactError,
    PredictionArtifactSource,
    download_prediction_rows,
)
from aitraf_api.schemas import (
    DemoPrediction,
    GroundTruth,
    TaskPrediction,
    TaskPredictions,
)


class PredictionRecord(TypedDict):
    video_id: str
    s3_path: str
    person: str
    trick: str
    execution_score: str
    label: str
    confidence: float


def load_demo_predictions(settings: Settings) -> list[DemoPrediction]:
    classification_rows = download_prediction_rows(
        PredictionArtifactSource(
            task="trick_classification",
            run_id=settings.classification_predictions_run_id,
        )
    )
    aqa_rows = download_prediction_rows(
        PredictionArtifactSource(
            task="trick_aqa",
            run_id=settings.aqa_predictions_run_id,
        )
    )
    return build_demo_predictions_response(
        classification_rows=classification_rows,
        aqa_rows=aqa_rows,
    )


def build_demo_predictions_response(
    *,
    classification_rows: list[PredictionRecord],
    aqa_rows: list[PredictionRecord],
) -> list[DemoPrediction]:
    aqa_by_video_id = {row["video_id"]: row for row in aqa_rows}

    predictions: list[DemoPrediction] = []
    for row in classification_rows:
        video_id = row["video_id"]
        aqa_row = aqa_by_video_id.get(video_id)
        if aqa_row is None:
            continue

        predictions.append(
            DemoPrediction(
                video_id=video_id,
                s3_path=row["s3_path"],
                person=row["person"],
                ground_truth=GroundTruth(
                    trick=row["trick"],
                    execution_score=row["execution_score"],
                ),
                predictions=TaskPredictions(
                    trick_classification=TaskPrediction(
                        label=row["label"],
                        confidence=row["confidence"],
                    ),
                    trick_aqa=TaskPrediction(
                        label=aqa_row["label"],
                        confidence=aqa_row["confidence"],
                    ),
                ),
            )
        )

    if not predictions:
        raise PredictionArtifactError(
            "Classification and AQA prediction artifacts have no matching video_id rows"
        )

    return predictions


__all__ = [
    "PredictionRecord",
    "build_demo_predictions_response",
    "load_demo_predictions",
]
