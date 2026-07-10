from __future__ import annotations

from pathlib import Path
from typing import Any

from aitraf_api.features.demo_predictions import thumbnails


class FakeS3Client:
    def __init__(self, *, existing: bool) -> None:
        self.existing = existing
        self.downloads: list[tuple[str, str]] = []
        self.uploads: list[tuple[str, str, dict[str, str]]] = []

    def list_objects_v2(
        self, *, Bucket: str, Prefix: str, MaxKeys: int
    ) -> dict[str, list[dict[str, str]]]:
        return {"Contents": [{"Key": Prefix}]} if self.existing else {}

    def put_object(self, *, Bucket: str, Key: str, Body: Any) -> None:
        assert Body.read() == b"jpeg"
        self.uploads.append((Bucket, Key))

    def download_file(self, bucket: str, key: str, path: str) -> None:
        self.downloads.append((bucket, key))
        Path(path).write_bytes(b"video")


def test_create_thumbnail_s3_path_uses_video_id() -> None:
    assert thumbnails.create_thumbnail_s3_path(
        "sample.mp4", bucket="aitraf"
    ) == "s3://aitraf/thumbnails/sample.jpg"


def test_find_rows_with_missing_thumbnails_returns_only_missing(
    monkeypatch: Any,
) -> None:
    client = FakeS3Client(existing=False)
    monkeypatch.setattr(thumbnails, "load_s3_settings", lambda **kwargs: _settings())
    monkeypatch.setattr(thumbnails, "build_s3_client", lambda settings: client)
    rows = [{"video_id": "sample.mp4", "s3_path": "s3://aitraf/clips/sample.mp4"}]

    assert thumbnails.find_rows_with_missing_thumbnails(
        rows,
        aws_endpoint_url="https://s3.example.test",
        aws_bucket="aitraf",
    ) == rows


def test_ensure_prediction_thumbnails_reuses_existing_object(
    monkeypatch: Any, tmp_path: Path
) -> None:
    client = FakeS3Client(existing=True)
    monkeypatch.setattr(thumbnails, "load_s3_settings", lambda **kwargs: _settings())
    monkeypatch.setattr(thumbnails, "build_s3_client", lambda settings: client)

    rows = thumbnails.ensure_prediction_thumbnails(
        [{"video_id": "sample.mp4", "s3_path": "s3://aitraf/clips/sample.mp4"}],
        local_videos={"s3://aitraf/clips/sample.mp4": tmp_path / "sample.mp4"},
        aws_endpoint_url="https://s3.example.test",
        aws_bucket="aitraf",
    )

    assert rows[0]["thumbnail_s3_path"] == "s3://aitraf/thumbnails/sample.jpg"
    assert client.downloads == []
    assert client.uploads == []


def test_ensure_prediction_thumbnails_generates_missing_object(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    client = FakeS3Client(existing=False)
    monkeypatch.setattr(thumbnails, "load_s3_settings", lambda **kwargs: _settings())
    monkeypatch.setattr(thumbnails, "build_s3_client", lambda settings: client)

    def extract(_video_path: Path, thumbnail_path: Path) -> None:
        thumbnail_path.write_bytes(b"jpeg")

    monkeypatch.setattr(thumbnails, "generate_thumbnail", extract)

    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"video")

    rows = thumbnails.ensure_prediction_thumbnails(
        [{"video_id": "sample.mp4", "s3_path": "s3://aitraf/clips/sample.mp4"}],
        local_videos={"s3://aitraf/clips/sample.mp4": video_path},
        aws_endpoint_url="https://s3.example.test",
        aws_bucket="aitraf",
    )

    assert rows[0]["thumbnail_s3_path"] == "s3://aitraf/thumbnails/sample.jpg"
    assert client.downloads == []
    assert client.uploads == [("aitraf", "thumbnails/sample.jpg")]


def _settings() -> Any:
    return type(
        "Settings",
        (),
        {"endpoint_url": "https://s3.example.test", "bucket": "aitraf"},
    )()
