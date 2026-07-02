from __future__ import annotations

import pytest

from aitraf_api.features.demo_predictions.artifacts import PredictionArtifactError
from aitraf_api.features.demo_predictions.loader import build_demo_predictions_response


def test_build_demo_predictions_response_matches_rows_by_video_id(
    classification_prediction_rows: list[dict],
    aqa_prediction_rows: list[dict],
) -> None:
    response = build_demo_predictions_response(
        classification_rows=classification_prediction_rows,
        aqa_rows=aqa_prediction_rows,
    )

    assert response[0].video_id == classification_prediction_rows[0]["video_id"]
    assert response[0].predictions.trick_classification.label == "mizou"
    assert response[0].predictions.trick_aqa.label == "3"


def test_build_demo_predictions_response_fails_without_matching_rows(
    classification_prediction_rows: list[dict],
    aqa_prediction_rows: list[dict],
) -> None:
    aqa_prediction_rows[0]["video_id"] = "different.mp4"

    with pytest.raises(PredictionArtifactError, match="no matching video_id rows"):
        build_demo_predictions_response(
            classification_rows=classification_prediction_rows,
            aqa_rows=aqa_prediction_rows,
        )
