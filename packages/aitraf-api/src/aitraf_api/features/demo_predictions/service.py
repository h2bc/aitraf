"""Demo prediction response construction."""

from __future__ import annotations

from typing import Any

from aitraf_api.features.demo_predictions.artifacts import PredictionArtifactError
from aitraf_api.features.demo_predictions.schemas import (
    DemoPrediction,
    GroundTruth,
    TaskPrediction,
    TaskPredictions,
)
from aitraf_api.features.demo_predictions.videos import VideoUrlPresigner


def build_demo_predictions_response(
    *,
    classification_rows: list[dict[str, Any]],
    aqa_rows: list[dict[str, Any]],
    presign_video_url: VideoUrlPresigner,
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
                video_url=presign_video_url(row["s3_path"]),
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


__all__ = ["build_demo_predictions_response"]
