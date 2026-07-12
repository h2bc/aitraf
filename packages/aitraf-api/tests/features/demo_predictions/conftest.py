from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from aitraf_api.config import Settings
from aitraf_api.features import router


@pytest.fixture()
def video_id() -> str:
    return "sample.mp4"


@pytest.fixture()
def settings() -> Settings:
    return Settings(
        api_token="test-token",
        classification_predictions_run_id="classification-run",
        aqa_predictions_run_id="aqa-run",
        aws_endpoint_url="https://s3.example.test",
        aws_bucket="aitraf",
        public_asset_bucket="aitraf-public",
    )


@pytest.fixture()
def classification_prediction_rows(video_id: str) -> list[dict]:
    return [
        {
            "video_id": video_id,
            "s3_path": f"s3://aitraf/clips/{video_id}",
            "video_url": f"https://s3.example.test/aitraf-public/videos/{video_id}",
            "thumbnail_url": f"https://s3.example.test/aitraf-public/thumbnails/{video_id.removesuffix('.mp4')}.jpg",
            "person": "person-a",
            "key_foot": "right",
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
            "key_foot": "right",
            "trick": "mizou",
            "execution_score": "3",
            "pred_id": 1,
            "label": "3",
            "confidence": 0.82,
        }
    ]


@pytest.fixture()
def client(
    settings: Settings,
    classification_prediction_rows: list[dict],
    aqa_prediction_rows: list[dict],
) -> TestClient:
    app = FastAPI()
    app.state.settings = settings
    app.state.classification_prediction_rows = classification_prediction_rows
    app.state.aqa_prediction_rows = aqa_prediction_rows
    app.include_router(router)
    return TestClient(app)
