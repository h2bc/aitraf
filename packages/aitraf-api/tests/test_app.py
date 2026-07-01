from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from aitraf_api import app as app_module
from aitraf_api.config import (
    DemoClipsConfig,
    Settings,
    TrickAssessmentConfig,
    TrickClassificationConfig,
)


class DummyAqaModel:
    run_params = {
        "num_clips": "2",
        "num_frames": "16",
        "train_sampling_dist": "uniform",
        "backbone": "backbone",
    }
    metadata = {
        "id2label": {"0": "1", "1": "2", "2": "3"},
        "target_column": "execution_score",
    }
    model = object()


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return Settings(
        api_token="test-token",
        device="cpu",
        clips_dir=tmp_path / "clips",
        classification=TrickClassificationConfig(
            model_uri="models:/classification/test",
            manifest_path=tmp_path / "classification.jsonl",
        ),
        aqa=TrickAssessmentConfig(
            model_uri="models:/aqa/test",
            manifest_path=tmp_path / "aqa.jsonl",
            features_dir=tmp_path / "features",
            frame_cache_dir=tmp_path / "frames",
            model_cache_dir=tmp_path / "models",
        ),
    )


def test_create_app_does_not_hydrate_demo_clips_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    calls = []
    _mock_model_loading(monkeypatch)
    monkeypatch.setattr(app_module, "hydrate_demo_clips", lambda value: calls.append(value))

    app_module.create_app(settings=settings)

    assert calls == [settings]


def test_create_app_hydrates_demo_clips_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    calls = []
    enabled = replace(
        settings,
        demo_clips=DemoClipsConfig(download_enabled=True),
    )
    _mock_model_loading(monkeypatch)
    monkeypatch.setattr(app_module, "hydrate_demo_clips", lambda value: calls.append(value))

    app_module.create_app(settings=enabled)

    assert calls == [enabled]


def _mock_model_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        app_module,
        "load_mlflow_transformers_model",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        app_module,
        "load_mlflow_torch_model",
        lambda *args, **kwargs: DummyAqaModel(),
    )
    monkeypatch.setattr(
        app_module,
        "load_huggingface_model",
        lambda *args, **kwargs: object(),
    )
