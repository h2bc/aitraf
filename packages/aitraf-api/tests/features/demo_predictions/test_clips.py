from typing import Any
from unittest.mock import Mock

import pytest

from aitraf_api.features.demo_predictions import clips


def _row(video_id: str = "sample.mp4", source: str | None = None) -> dict[str, Any]:
    return {"video_id": video_id, "s3_path": source or f"s3://aitraf/clips/{video_id}"}


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> Mock:
    client = Mock()
    settings = Mock(endpoint_url="https://s3.test", bucket="aitraf")
    monkeypatch.setattr(clips, "load_s3_settings", lambda **kwargs: settings)
    monkeypatch.setattr(clips, "build_s3_client", lambda settings: client)
    return client


def _prepare(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return clips.prepare_public_demo_clips(
        rows,
        aws_endpoint_url="https://s3.test",
        source_bucket="aitraf",
        public_bucket="public",
    )


def test_prepare_reuses_existing_clip(
    monkeypatch: pytest.MonkeyPatch, client: Mock
) -> None:
    monkeypatch.setattr(clips, "object_exists", lambda *args, **kwargs: True)
    copy = Mock()
    upload = Mock()
    monkeypatch.setattr(clips, "copy_s3_object", copy)
    monkeypatch.setattr(clips, "_upload_thumbnail", upload)

    prepared = _prepare([_row()])

    assert prepared[0]["video_url"] == "https://s3.test/public/videos/sample.mp4"
    assert prepared[0]["thumbnail_url"] == "https://s3.test/public/thumbnails/sample.jpg"
    copy.assert_not_called()
    upload.assert_not_called()


def test_prepare_publishes_missing_clip(
    monkeypatch: pytest.MonkeyPatch, client: Mock
) -> None:
    existing = iter([True, False, False])
    monkeypatch.setattr(clips, "object_exists", lambda *args, **kwargs: next(existing))
    copy = Mock()
    create = Mock(return_value=Mock())
    upload = Mock()
    monkeypatch.setattr(clips, "copy_s3_object", copy)
    monkeypatch.setattr(clips, "_create_thumbnail", create)
    monkeypatch.setattr(clips, "_upload_thumbnail", upload)

    _prepare([_row()])

    copy.assert_called_once()
    create.assert_called_once()
    upload.assert_called_once()


def test_prepare_rejects_missing_source(
    monkeypatch: pytest.MonkeyPatch, client: Mock
) -> None:
    monkeypatch.setattr(clips, "object_exists", lambda *args, **kwargs: False)
    with pytest.raises(clips.DemoClipError, match="Missing source video"):
        _prepare([_row()])
