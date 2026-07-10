import pytest

from aitraf_api.features.demo_predictions.artifacts import PredictionArtifactError
from aitraf_api.features.demo_predictions.service import match_prediction_rows


def test_match_prediction_rows_filters_before_video_download() -> None:
    classification = [
        {"video_id": "matched.mp4"},
        {"video_id": "classification-only.mp4"},
    ]
    aqa = [{"video_id": "matched.mp4"}, {"video_id": "aqa-only.mp4"}]

    assert match_prediction_rows(classification, aqa) == (
        [{"video_id": "matched.mp4"}],
        [{"video_id": "matched.mp4"}],
    )


def test_match_prediction_rows_rejects_empty_intersection() -> None:
    with pytest.raises(PredictionArtifactError, match="no matching video_id"):
        match_prediction_rows(
            [{"video_id": "classification.mp4"}],
            [{"video_id": "aqa.mp4"}],
        )
