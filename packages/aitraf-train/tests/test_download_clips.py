from __future__ import annotations

from io import BytesIO
from pathlib import Path

from aitraf_train.preparation.data_ops import download_clips as download_module
from aitraf_train.preparation.data_ops.download_clips import (
    ClipDownloadConfig,
    build_clip_download_requests_from_labels,
    download_clips,
)


class FakeS3Client:
    def __init__(self) -> None:
        self.downloads: list[tuple[str, str]] = []

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, BytesIO]:
        self.downloads.append((Bucket, Key))
        return {"Body": BytesIO(f"{Bucket}/{Key}".encode())}


def test_build_clip_download_requests_from_labels_preserves_destination_paths(
    tmp_path: Path,
) -> None:
    labels_path = tmp_path / "labels.jsonl"
    labels_path.write_text(
        '{"video":"s3://aitraf/clips/a.mp4"}\n'
        '{"video":"s3://aitraf/clips/nested/b.mp4"}\n',
        encoding="utf-8",
    )

    requests = build_clip_download_requests_from_labels(labels_path)

    assert [
        (request.video_id, request.source_uri, request.relative_path)
        for request in requests
    ] == [
        ("a.mp4", "s3://aitraf/clips/a.mp4", Path("a.mp4")),
        ("nested/b.mp4", "s3://aitraf/clips/nested/b.mp4", Path("nested/b.mp4")),
    ]


def test_download_clips_uses_core_requests_and_skips_existing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    labels_path = tmp_path / "labels.jsonl"
    labels_path.write_text('{"video":"s3://aitraf/clips/a.mp4"}\n', encoding="utf-8")
    output_dir = tmp_path / "clips"
    output_dir.mkdir()
    (output_dir / "a.mp4").write_text("existing", encoding="utf-8")
    client = FakeS3Client()
    monkeypatch.setattr(download_module, "build_s3_client", lambda settings: client)
    monkeypatch.setattr(
        download_module,
        "load_s3_settings",
        lambda require_bucket: object(),
    )

    download_clips(
        ClipDownloadConfig(
            labels_path=labels_path,
            output_dir=output_dir,
            force=False,
        )
    )

    assert (output_dir / "a.mp4").read_text(encoding="utf-8") == "existing"
    assert client.downloads == []


def test_download_clips_force_redownloads_existing_file(
    monkeypatch,
    tmp_path: Path,
) -> None:
    labels_path = tmp_path / "labels.jsonl"
    labels_path.write_text('{"video":"s3://aitraf/clips/a.mp4"}\n', encoding="utf-8")
    output_dir = tmp_path / "clips"
    output_dir.mkdir()
    (output_dir / "a.mp4").write_text("existing", encoding="utf-8")
    client = FakeS3Client()
    monkeypatch.setattr(download_module, "build_s3_client", lambda settings: client)
    monkeypatch.setattr(
        download_module,
        "load_s3_settings",
        lambda require_bucket: object(),
    )

    download_clips(
        ClipDownloadConfig(
            labels_path=labels_path,
            output_dir=output_dir,
            force=True,
        )
    )

    assert (output_dir / "a.mp4").read_text(encoding="utf-8") == "aitraf/clips/a.mp4"
    assert client.downloads == [("aitraf", "clips/a.mp4")]
