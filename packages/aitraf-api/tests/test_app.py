from typing import Any

import pytest
from fastapi.testclient import TestClient

from aitraf_api import app as app_module
from aitraf_api.config import Settings
from aitraf_api.features.demo_predictions.clips import DemoClipError


def _settings() -> Settings:
    return Settings(
        "token",
        "classification",
        "aqa",
        "https://s3.test",
        "private",
        "public",
        "redis://localhost:6379/0",
    )


class RecordingCounter:
    def __init__(self) -> None:
        self.validate_calls = 0
        self.close_calls = 0

    async def validate(self) -> None:
        self.validate_calls += 1

    async def increment(self) -> int:
        return 1

    async def aclose(self) -> None:
        self.close_calls += 1


def test_create_app_prepares_only_matched_classification_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    classification = [
        {"video_id": "matched.mp4", "s3_path": "s3://private/matched.mp4"},
        {"video_id": "unmatched.mp4", "s3_path": "s3://private/unmatched.mp4"},
    ]
    aqa = [{"video_id": "matched.mp4"}]
    captured: list[dict[str, Any]] = []
    monkeypatch.setattr(
        app_module,
        "download_demo_prediction_rows",
        lambda settings: (classification, aqa),
    )

    def prepare(rows: list[dict[str, Any]], **kwargs: Any) -> list[dict[str, Any]]:
        captured.extend(rows)
        return [
            {
                **row,
                "video_url": "https://s3.test/public/videos/matched.mp4",
                "thumbnail_url": "https://s3.test/public/thumbnails/matched.jpg",
            }
            for row in rows
        ]

    monkeypatch.setattr(app_module, "prepare_public_demo_clips", prepare)
    created = app_module.create_app(
        settings=_settings(), visitor_counter=RecordingCounter()
    )
    assert [row["video_id"] for row in captured] == ["matched.mp4"]
    assert created.state.classification_prediction_rows[0]["video_url"].startswith(
        "https://s3.test/public/"
    )


def test_app_validates_and_closes_visitor_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        app_module,
        "download_demo_prediction_rows",
        lambda settings: ([{"video_id": "matched.mp4"}], [{"video_id": "matched.mp4"}]),
    )
    monkeypatch.setattr(
        app_module,
        "prepare_public_demo_clips",
        lambda rows, **kwargs: rows,
    )
    counter = RecordingCounter()
    app = app_module.create_app(settings=_settings(), visitor_counter=counter)

    with TestClient(app):
        assert counter.validate_calls == 1
        assert counter.close_calls == 0

    assert counter.close_calls == 1


def test_create_app_fails_when_pose_data_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A selected prediction without usable poses must block startup entirely.

    Readiness with a non-pose asset is the failure this feature must never
    produce, so the error propagates instead of degrading to the source clip.
    """
    monkeypatch.setattr(
        app_module,
        "download_demo_prediction_rows",
        lambda settings: ([{"video_id": "matched.mp4"}], [{"video_id": "matched.mp4"}]),
    )

    def prepare(rows: list[dict[str, Any]], **kwargs: Any) -> list[dict[str, Any]]:
        raise DemoClipError("Missing pose artifact for 'matched.mp4'")

    monkeypatch.setattr(app_module, "prepare_public_demo_clips", prepare)

    with pytest.raises(DemoClipError, match="Missing pose artifact"):
        app_module.create_app(settings=_settings(), visitor_counter=RecordingCounter())
