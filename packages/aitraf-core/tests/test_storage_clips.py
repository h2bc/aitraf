from __future__ import annotations

from pathlib import Path

import pytest

from aitraf_core.storage import clips as clips_module
from aitraf_core.storage.clips import ClipDownloadRequest, download_clip, download_clips


class FakeS3Client:
    def __init__(self) -> None:
        self.downloads: list[tuple[str, str, str]] = []

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        self.downloads.append((bucket, key, filename))
        Path(filename).write_text(f"{bucket}/{key}", encoding="utf-8")


def test_download_clip_downloads_missing_file(tmp_path: Path) -> None:
    client = FakeS3Client()
    request = ClipDownloadRequest(
        video_id="sample.mp4",
        source_uri="s3://aitraf/clips/sample.mp4",
    )

    result = download_clip(request, clips_dir=tmp_path, s3_client=client)

    assert result.status == "downloaded"
    assert result.destination_path == tmp_path / "sample.mp4"
    assert result.destination_path.read_text(encoding="utf-8") == "aitraf/clips/sample.mp4"
    assert client.downloads == [
        ("aitraf", "clips/sample.mp4", str(tmp_path / "sample.mp4"))
    ]


def test_download_clip_skips_existing_file(tmp_path: Path) -> None:
    client = FakeS3Client()
    existing = tmp_path / "sample.mp4"
    existing.write_text("already here", encoding="utf-8")
    request = ClipDownloadRequest(
        video_id="sample.mp4",
        source_uri="s3://aitraf/clips/sample.mp4",
    )

    result = download_clip(request, clips_dir=tmp_path, s3_client=client)

    assert result.status == "skipped-existing"
    assert existing.read_text(encoding="utf-8") == "already here"
    assert client.downloads == []


def test_download_clip_force_overwrites_existing_file(tmp_path: Path) -> None:
    client = FakeS3Client()
    existing = tmp_path / "sample.mp4"
    existing.write_text("old", encoding="utf-8")
    request = ClipDownloadRequest(
        video_id="sample.mp4",
        source_uri="s3://aitraf/clips/sample.mp4",
    )

    result = download_clip(request, clips_dir=tmp_path, force=True, s3_client=client)

    assert result.status == "downloaded"
    assert existing.read_text(encoding="utf-8") == "aitraf/clips/sample.mp4"
    assert client.downloads


def test_download_clip_rejects_unsafe_destination(tmp_path: Path) -> None:
    request = ClipDownloadRequest(
        video_id="../sample.mp4",
        source_uri="s3://aitraf/clips/sample.mp4",
    )

    with pytest.raises(ValueError, match="relative"):
        download_clip(request, clips_dir=tmp_path, s3_client=FakeS3Client())


def test_download_clips_downloads_all_requests(tmp_path: Path) -> None:
    client = FakeS3Client()
    requests = [
        ClipDownloadRequest("a.mp4", "s3://aitraf/clips/a.mp4"),
        ClipDownloadRequest("b.mp4", "s3://aitraf/clips/b.mp4"),
    ]

    results = download_clips(requests, clips_dir=tmp_path, s3_client=client)

    assert [result.status for result in results] == ["downloaded", "downloaded"]
    assert [path.name for path in tmp_path.iterdir()] == ["a.mp4", "b.mp4"]


def test_download_clips_reuses_lazy_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeS3Client()
    build_calls = 0

    def fake_build_s3_client(settings):
        nonlocal build_calls
        build_calls += 1
        return client

    monkeypatch.setattr(clips_module, "load_s3_settings", lambda **_: object())
    monkeypatch.setattr(clips_module, "build_s3_client", fake_build_s3_client)
    requests = [
        ClipDownloadRequest("a.mp4", "s3://aitraf/clips/a.mp4"),
        ClipDownloadRequest("b.mp4", "s3://aitraf/clips/b.mp4"),
    ]

    results = download_clips(requests, clips_dir=tmp_path)

    assert [result.status for result in results] == ["downloaded", "downloaded"]
    assert build_calls == 1
