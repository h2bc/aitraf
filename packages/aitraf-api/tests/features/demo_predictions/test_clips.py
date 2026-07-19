from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from aitraf_api.features.demo_predictions import clips

SOURCE_VIDEO = "clips/sample.mp4"
SOURCE_POSES = "poses/sample.npz"
PUBLIC_VIDEO = "videos/sample.mp4"
PUBLIC_THUMBNAIL = "thumbnails/sample.jpg"


def _row(video_id: str = "sample.mp4", source: str | None = None) -> dict[str, Any]:
    return {"video_id": video_id, "s3_path": source or f"s3://aitraf/clips/{video_id}"}


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> Mock:
    client = Mock()
    settings = Mock(endpoint_url="https://s3.test", bucket="aitraf")
    monkeypatch.setattr(clips, "load_s3_settings", lambda **kwargs: settings)
    monkeypatch.setattr(clips, "build_s3_client", lambda settings: client)
    return client


@pytest.fixture()
def render(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Stub the download/render chain, writing a placeholder rendered file."""

    def _download(uri: str, destination: Path, **kwargs: Any) -> None:
        destination.write_bytes(b"payload")

    render = Mock()
    monkeypatch.setattr(clips, "download_s3_uri", _download)
    monkeypatch.setattr(clips, "load_pose_artifact", Mock())
    monkeypatch.setattr(clips, "render_pose_video", render)
    monkeypatch.setattr(
        clips, "generate_thumbnail", lambda source, dest: dest.write_bytes(b"jpeg")
    )
    return render


def _prepare(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return clips.prepare_public_demo_clips(
        rows,
        aws_endpoint_url="https://s3.test",
        source_bucket="aitraf",
        public_bucket="public",
    )


def _exists(*present: str):
    """Return an object_exists stub that reports only `present` keys."""

    def _stub(_client: Any, *, bucket: str, key: str) -> bool:
        return key in present

    return _stub


def test_prepare_reuses_existing_assets(
    monkeypatch: pytest.MonkeyPatch, client: Mock, render: Mock
) -> None:
    monkeypatch.setattr(clips, "object_exists", _exists(PUBLIC_VIDEO, PUBLIC_THUMBNAIL))
    upload = Mock()
    monkeypatch.setattr(clips, "_upload_file", upload)

    prepared = _prepare([_row()])

    assert prepared[0]["video_url"] == f"https://s3.test/public/{PUBLIC_VIDEO}"
    assert prepared[0]["thumbnail_url"] == f"https://s3.test/public/{PUBLIC_THUMBNAIL}"
    render.assert_not_called()
    upload.assert_not_called()


def test_prepare_renders_and_publishes_both_assets(
    monkeypatch: pytest.MonkeyPatch, client: Mock, render: Mock
) -> None:
    monkeypatch.setattr(clips, "object_exists", _exists(SOURCE_VIDEO, SOURCE_POSES))
    upload = Mock()
    monkeypatch.setattr(clips, "_upload_file", upload)

    _prepare([_row()])

    render.assert_called_once()
    assert [call.kwargs["key"] for call in upload.call_args_list] == [
        PUBLIC_VIDEO,
        PUBLIC_THUMBNAIL,
    ]
    assert [call.kwargs["content_type"] for call in upload.call_args_list] == [
        "video/mp4",
        "image/jpeg",
    ]


def test_prepare_reuses_published_video_for_missing_thumbnail(
    monkeypatch: pytest.MonkeyPatch, client: Mock, render: Mock
) -> None:
    monkeypatch.setattr(
        clips, "object_exists", _exists(PUBLIC_VIDEO, SOURCE_VIDEO, SOURCE_POSES)
    )
    upload = Mock()
    monkeypatch.setattr(clips, "_upload_file", upload)

    _prepare([_row()])

    render.assert_not_called()
    assert [call.kwargs["key"] for call in upload.call_args_list] == [PUBLIC_THUMBNAIL]


def test_prepare_rejects_missing_source_video(
    monkeypatch: pytest.MonkeyPatch, client: Mock, render: Mock
) -> None:
    monkeypatch.setattr(clips, "object_exists", _exists())

    with pytest.raises(clips.DemoClipError, match="Missing source video"):
        _prepare([_row()])


def test_prepare_rejects_missing_pose_artifact(
    monkeypatch: pytest.MonkeyPatch, client: Mock, render: Mock
) -> None:
    monkeypatch.setattr(clips, "object_exists", _exists(SOURCE_VIDEO))

    with pytest.raises(clips.DemoClipError, match="Missing pose artifact"):
        _prepare([_row()])


def test_prepare_propagates_invalid_pose_artifact(
    monkeypatch: pytest.MonkeyPatch, client: Mock, render: Mock
) -> None:
    monkeypatch.setattr(clips, "object_exists", _exists(SOURCE_VIDEO, SOURCE_POSES))
    monkeypatch.setattr(clips, "_upload_file", Mock())
    monkeypatch.setattr(
        clips, "load_pose_artifact", Mock(side_effect=ValueError("bad artifact"))
    )

    with pytest.raises(clips.DemoClipError, match="Failed to publish demo assets"):
        _prepare([_row()])


def test_prepare_rejects_invalid_video_id(client: Mock, render: Mock) -> None:
    with pytest.raises(clips.DemoClipError, match="Invalid video_id"):
        _prepare([_row(video_id="nested/sample.mp4")])


def test_prepare_rejects_foreign_source_bucket(client: Mock, render: Mock) -> None:
    with pytest.raises(clips.DemoClipError, match="Expected source bucket"):
        _prepare([_row(source="s3://other/clips/sample.mp4")])
