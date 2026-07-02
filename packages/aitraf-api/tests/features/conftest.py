from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from aitraf_api.config import Settings
from aitraf_api.features import router
from aitraf_api.schemas import (
    DemoPrediction,
    GroundTruth,
    TaskPrediction,
    TaskPredictions,
)


@pytest.fixture()
def video_id() -> str:
    return "sample.mp4"


@pytest.fixture()
def settings() -> Settings:
    return Settings(
        api_token="test-token",
        classification_predictions_run_id="classification-run",
        aqa_predictions_run_id="aqa-run",
    )


@pytest.fixture()
def demo_predictions_response(video_id: str) -> list[DemoPrediction]:
    return [
        DemoPrediction(
            video_id=video_id,
            s3_path=f"s3://aitraf/clips/{video_id}",
            person="person-a",
            ground_truth=GroundTruth(trick="mizou", execution_score="3"),
            predictions=TaskPredictions(
                trick_classification=TaskPrediction(
                    label="mizou",
                    confidence=0.91,
                ),
                trick_aqa=TaskPrediction(
                    label="3",
                    confidence=0.82,
                ),
            ),
        )
    ]


@pytest.fixture()
def client(
    settings: Settings,
    demo_predictions_response: list[DemoPrediction],
) -> TestClient:
    app = FastAPI()
    app.state.settings = settings
    app.state.demo_predictions = demo_predictions_response
    app.include_router(router)
    return TestClient(app)
