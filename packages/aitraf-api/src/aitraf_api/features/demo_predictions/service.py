"""Build demo prediction responses from downloaded artifact rows."""

from __future__ import annotations

from typing import Any

from aitraf_api.features.demo_predictions.artifacts import PredictionArtifactError
from aitraf_api.schemas import (
    DemoPrediction,
    GroundTruth,
    TaskPrediction,
    TaskPredictions,
)

PredictionRecord = dict[str, Any]


def build_demo_predictions_response(
    *,
    classification_rows: list[PredictionRecord],
    aqa_rows: list[PredictionRecord],
) -> list[DemoPrediction]:
    aqa_by_video_id = {str(row["video_id"]): row for row in aqa_rows}

    predictions: list[DemoPrediction] = []
    for row in classification_rows:
        video_id = str(row["video_id"])
        aqa_row = aqa_by_video_id.get(video_id)
        if aqa_row is None:
            continue

        predictions.append(
            DemoPrediction(
                video_id=video_id,
                s3_path=str(row["s3_path"]),
                person=str(row["person"]),
                ground_truth=GroundTruth(
                    trick=str(row["trick"]),
                    execution_score=str(row["execution_score"]),
                ),
                predictions=TaskPredictions(
                    trick_classification=TaskPrediction(
                        label=str(row["label"]),
                        confidence=float(row["confidence"]),
                    ),
                    trick_aqa=TaskPrediction(
                        label=str(aqa_row["label"]),
                        confidence=float(aqa_row["confidence"]),
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
    "build_demo_predictions_response",
]
