from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from aitraf_api.app import create_app
from aitraf_api.config import Settings, TrickAssessmentConfig, TrickClassificationConfig
from aitraf_core.pre_processing import (
    video_feature_cache_dir,
    video_feature_cache_path,
)


@pytest.fixture()
def api_token() -> str:
    return "test-token"


@pytest.fixture()
def auth_headers(api_token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_token}"}


@pytest.fixture()
def manifest_paths(tmp_path: Path) -> tuple[Path, Path]:
    classification = tmp_path / "classification.jsonl"
    aqa = tmp_path / "aqa.jsonl"
    shared = {
        "video_id": "shared.mp4",
        "s3_path": "s3://aitraf/clips/shared.mp4",
        "trick": "top-soul",
        "key_foot": "right",
        "person": "Justas",
        "execution_score": "3",
    }
    classification_only = {
        "video_id": "classification-only.mp4",
        "trick": "soul",
        "person": "Max",
        "execution_score": "2",
    }
    aqa_only = {
        "video_id": "aqa-only.mp4",
        "trick": "mizou",
        "person": "Rytis",
        "execution_score": "1",
    }
    _write_jsonl(classification, [shared, classification_only])
    _write_jsonl(aqa, [shared, aqa_only])
    return classification, aqa


@pytest.fixture()
def settings(api_token: str, manifest_paths: tuple[Path, Path], tmp_path: Path) -> Settings:
    classification_path, aqa_path = manifest_paths
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    (clips_dir / "shared.mp4").touch()
    features_dir = tmp_path / "video_mae_features"
    feature_cache_dir = video_feature_cache_dir(
        features_dir=features_dir,
        backbone="test/backbone",
        num_clips=2,
        sample_frames=1,
        sampling_dist="uniform",
    )
    feature_path = video_feature_cache_path(
        feature_cache_dir=feature_cache_dir,
        video_id="shared.mp4",
    )
    feature_path.parent.mkdir(parents=True)
    torch.save({"features": torch.ones(2, 4)}, feature_path)
    return Settings(
        api_token=api_token,
        clips_dir=clips_dir,
        classification=TrickClassificationConfig(
            model_uri="models:/aitraf-trick-classification@infant",
            manifest_path=classification_path,
        ),
        aqa=TrickAssessmentConfig(
            model_uri="models:/aitraf-trick-aqa@infant",
            manifest_path=aqa_path,
            features_dir=features_dir,
            frame_cache_dir=tmp_path / "frame_cache",
            model_cache_dir=tmp_path / "models",
        ),
    )


@pytest.fixture()
def client(settings: Settings) -> TestClient:
    app = create_app(settings=settings)
    return TestClient(app)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
