from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from aitraf_api.app import create_app
from aitraf_api.config import RegisteredModelReference, Settings


@dataclass(frozen=True)
class StubLoadedModel:
    model_uri: str


class StubPredictVideo:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def load_model(
        self,
        model_uri: str,
        tracking_uri: str | None,
    ) -> StubLoadedModel:
        _ = tracking_uri
        return StubLoadedModel(model_uri=model_uri)

    def predict(
        self,
        *,
        loaded_model: StubLoadedModel,
        video_id: str,
        local_clips_dir: Path,
    ) -> tuple[str, float]:
        _ = local_clips_dir
        self.calls.append((loaded_model.model_uri, video_id))
        if "classification" in loaded_model.model_uri:
            return "top-soul", 0.91
        return "3", 0.72


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
    return Settings(
        api_token=api_token,
        mlflow_tracking_uri="http://mlflow.test",
        clips_dir=clips_dir,
        classification=RegisteredModelReference(
            model_uri="models:/aitraf-trick-classification@infant",
            model_kind="video_mae",
            manifest_path=classification_path,
            ground_truth_field="trick",
        ),
        aqa=RegisteredModelReference(
            model_uri="models:/aitraf-trick-aqa@infant",
            model_kind="ordinal",
            manifest_path=aqa_path,
            ground_truth_field="execution_score",
        ),
    )


@pytest.fixture()
def predict_video() -> StubPredictVideo:
    return StubPredictVideo()


@pytest.fixture()
def client(settings: Settings, predict_video: StubPredictVideo) -> TestClient:
    app = create_app(
        settings=settings,
        load_model=predict_video.load_model,
        predict_video=predict_video.predict,
    )
    return TestClient(app)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
