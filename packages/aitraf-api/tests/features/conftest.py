from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from aitraf_api.config import (
    Settings,
    TrickAssessmentConfig,
    TrickAssessmentPreProcessingConfig,
    TrickClassificationConfig,
)
from aitraf_api.features import router


@pytest.fixture()
def video_id() -> str:
    return "sample.mp4"


@pytest.fixture()
def settings(tmp_path: Path, video_id: str) -> Settings:
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    (clips_dir / video_id).touch()

    classification_manifest = tmp_path / "trick_classification.jsonl"
    classification_manifest.write_text(
        json.dumps(
            {
                "video_id": video_id,
                "s3_path": f"s3://aitraf/clips/{video_id}",
                "trick": "mizou",
                "execution_score": "2",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    aqa_manifest = tmp_path / "trick_aqa.jsonl"
    aqa_manifest.write_text(
        json.dumps(
            {
                "video_id": video_id,
                "s3_path": f"s3://aitraf/clips/{video_id}",
                "trick": "mizou",
                "execution_score": "3",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    return Settings(
        api_token="test-token",
        device="cpu",
        clips_dir=clips_dir,
        classification=TrickClassificationConfig(
            model_uri="models:/classification/test",
            manifest_path=classification_manifest,
        ),
        aqa=TrickAssessmentConfig(
            model_uri="models:/aqa/test",
            manifest_path=aqa_manifest,
            features_dir=tmp_path / "features",
            frame_cache_dir=tmp_path / "frames",
            model_cache_dir=tmp_path / "models",
        ),
    )


@pytest.fixture()
def client(settings: Settings) -> TestClient:
    app = FastAPI()
    app.state.settings = settings
    app.state.classification_model = object()
    app.state.aqa_model = object()
    app.state.aqa_feature_extractor = object()
    app.state.aqa_pre_processing = TrickAssessmentPreProcessingConfig(
        backbone="backbone",
        num_clips=2,
        sample_frames=16,
        sampling_dist="uniform",
        feature_cache_dir=settings.aqa.features_dir,
        id2label={"0": "1", "1": "2", "2": "3"},
    )
    app.include_router(router)
    return TestClient(app)
