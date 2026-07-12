from typing import Any

import pytest

from aitraf_api import app as app_module
from aitraf_api.config import Settings


def _settings() -> Settings:
    return Settings("token", "classification", "aqa", "https://s3.test", "private", "public")


def test_create_app_prepares_only_matched_classification_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    classification = [
        {"video_id": "matched.mp4", "s3_path": "s3://private/matched.mp4"},
        {"video_id": "unmatched.mp4", "s3_path": "s3://private/unmatched.mp4"},
    ]
    aqa = [{"video_id": "matched.mp4"}]
    captured: list[dict[str, Any]] = []
    monkeypatch.setattr(app_module, "download_demo_prediction_rows", lambda settings: (classification, aqa))

    def prepare(rows: list[dict[str, Any]], **kwargs: Any) -> list[dict[str, Any]]:
        captured.extend(rows)
        return [{**row, "video_url": "https://s3.test/public/videos/matched.mp4", "thumbnail_url": "https://s3.test/public/thumbnails/matched.jpg"} for row in rows]

    monkeypatch.setattr(app_module, "prepare_public_demo_clips", prepare)
    created = app_module.create_app(settings=_settings())
    assert [row["video_id"] for row in captured] == ["matched.mp4"]
    assert created.state.classification_prediction_rows[0]["video_url"].startswith(
        "https://s3.test/public/"
    )
