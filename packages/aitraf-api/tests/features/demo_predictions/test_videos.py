from io import BytesIO
from pathlib import Path
from typing import Any

import pytest

from aitraf_api.features.demo_predictions import videos


class FakeS3Client:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, BytesIO]:
        if self.fail:
            raise RuntimeError("download failed")
        return {"Body": BytesIO(b"video")}


def test_download_prediction_videos_returns_local_paths(
    monkeypatch: Any, tmp_path: Path
) -> None:
    _patch_client(monkeypatch, FakeS3Client())

    local = videos.download_prediction_videos(
        [{"video_id": "sample.mp4", "s3_path": "s3://aitraf/clips/sample.mp4"}],
        aws_endpoint_url="https://s3.example.test",
        aws_bucket="aitraf",
        directory=tmp_path,
    )

    assert local["s3://aitraf/clips/sample.mp4"].read_bytes() == b"video"


def test_failed_video_download_raises_before_thumbnail_phase(
    monkeypatch: Any, tmp_path: Path
) -> None:
    _patch_client(monkeypatch, FakeS3Client(fail=True))

    with pytest.raises(videos.VideoDownloadError, match="Failed to download"):
        videos.download_prediction_videos(
            [{"video_id": "sample.mp4", "s3_path": "s3://aitraf/clips/sample.mp4"}],
            aws_endpoint_url="https://s3.example.test",
            aws_bucket="aitraf",
            directory=tmp_path,
        )


def _patch_client(monkeypatch: Any, client: FakeS3Client) -> None:
    settings = type(
        "Settings",
        (),
        {"endpoint_url": "https://s3.example.test", "bucket": "aitraf"},
    )()
    monkeypatch.setattr(videos, "load_s3_settings", lambda **kwargs: settings)
    monkeypatch.setattr(videos, "build_s3_client", lambda value: client)
