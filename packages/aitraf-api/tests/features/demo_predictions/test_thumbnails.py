from pathlib import Path
from typing import Any

import pytest

from aitraf_api.features.demo_predictions import thumbnails


def test_generate_thumbnail_preserves_source_dimensions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    command: list[str] = []

    def run(args: list[str], **_kwargs: Any) -> None:
        command.extend(args)
        Path(args[-1]).write_bytes(b"jpeg")

    monkeypatch.setattr(thumbnails.subprocess, "run", run)
    thumbnails.generate_thumbnail(tmp_path / "sample.mp4", tmp_path / "thumbnail.jpg")
    assert "-vf" not in command
    assert not any(argument.startswith("scale=") for argument in command)


def test_generate_thumbnail_fails_when_ffmpeg_creates_no_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(thumbnails.subprocess, "run", lambda *args, **kwargs: None)
    with pytest.raises(thumbnails.ThumbnailError, match="did not create"):
        thumbnails.generate_thumbnail(
            tmp_path / "sample.mp4", tmp_path / "thumbnail.jpg"
        )
