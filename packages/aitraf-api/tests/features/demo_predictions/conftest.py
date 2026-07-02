from __future__ import annotations

import pytest


@pytest.fixture()
def classification_prediction_rows(video_id: str) -> list[dict]:
    return [
        {
            "video_id": video_id,
            "s3_path": f"s3://aitraf/clips/{video_id}",
            "person": "person-a",
            "trick": "mizou",
            "execution_score": "3",
            "pred_id": 2,
            "label": "mizou",
            "confidence": 0.91,
        }
    ]


@pytest.fixture()
def aqa_prediction_rows(video_id: str) -> list[dict]:
    return [
        {
            "video_id": video_id,
            "s3_path": f"s3://aitraf/clips/{video_id}",
            "person": "person-a",
            "trick": "mizou",
            "execution_score": "3",
            "pred_id": 1,
            "label": "3",
            "confidence": 0.82,
        }
    ]
