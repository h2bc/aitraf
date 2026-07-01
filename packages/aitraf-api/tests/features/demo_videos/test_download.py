from __future__ import annotations

from pathlib import Path

import pytest

from aitraf_api.config import DemoClipsConfig, Settings
from aitraf_api.features.demo_videos import download as download_module
from aitraf_api.features.demo_videos.download import (
    build_demo_clip_download_requests,
    hydrate_demo_clips,
)


def test_build_demo_clip_download_requests_uses_selected_demo_rows() -> None:
    classification_rows = [
        {"video_id": "a.mp4", "s3_path": "s3://aitraf/clips/a.mp4"},
        {"video_id": "b.mp4", "s3_path": "s3://aitraf/clips/b.mp4"},
        {"video_id": "a.mp4", "s3_path": "s3://aitraf/clips/a-duplicate.mp4"},
    ]
    aqa_rows = [
        {"video_id": "a.mp4", "s3_path": "s3://aitraf/clips/a.mp4"},
        {"video_id": "c.mp4", "s3_path": "s3://aitraf/clips/c.mp4"},
    ]

    requests = build_demo_clip_download_requests(
        classification_rows=classification_rows,
        aqa_rows=aqa_rows,
    )

    assert [(request.video_id, request.source_uri) for request in requests] == [
        ("a.mp4", "s3://aitraf/clips/a.mp4")
    ]


def test_build_demo_clip_download_requests_requires_source_uri() -> None:
    with pytest.raises(RuntimeError, match="s3_path"):
        build_demo_clip_download_requests(
            classification_rows=[{"video_id": "a.mp4"}],
            aqa_rows=[{"video_id": "a.mp4"}],
        )


def test_build_demo_clip_download_requests_rejects_invalid_source_uri() -> None:
    with pytest.raises(ValueError, match="Unsupported clip URI"):
        build_demo_clip_download_requests(
            classification_rows=[
                {"video_id": "a.mp4", "s3_path": "https://example.test/a.mp4"}
            ],
            aqa_rows=[{"video_id": "a.mp4"}],
        )


def test_build_demo_clip_download_requests_requires_matching_demo_rows() -> None:
    with pytest.raises(RuntimeError, match="no matching videos"):
        build_demo_clip_download_requests(
            classification_rows=[{"video_id": "a.mp4"}],
            aqa_rows=[{"video_id": "b.mp4"}],
        )


def test_hydrate_demo_clips_does_nothing_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    calls = []
    monkeypatch.setattr(download_module, "download_clips", lambda *args, **kwargs: calls.append(args))

    hydrate_demo_clips(settings)

    assert calls == []


def test_hydrate_demo_clips_downloads_selected_rows_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    video_id: str,
) -> None:
    calls = []
    enabled = _with_demo_clips(settings, download_enabled=True)
    monkeypatch.setattr(
        download_module,
        "download_clips",
        lambda requests, **kwargs: calls.append((requests, kwargs)),
    )

    hydrate_demo_clips(enabled)

    assert len(calls) == 1
    requests, kwargs = calls[0]
    assert [(request.video_id, request.source_uri) for request in requests] == [
        (video_id, f"s3://aitraf/clips/{video_id}")
    ]
    assert kwargs == {"clips_dir": settings.clips_dir, "force": False}


def test_hydrate_demo_clips_propagates_download_failures(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    enabled = _with_demo_clips(settings, download_enabled=True)

    def fail_download(*args, **kwargs):
        raise RuntimeError("download failed")

    monkeypatch.setattr(download_module, "download_clips", fail_download)

    with pytest.raises(RuntimeError, match="download failed"):
        hydrate_demo_clips(enabled)


def _with_demo_clips(
    settings: Settings,
    *,
    download_enabled: bool,
    force_download: bool = False,
) -> Settings:
    return Settings(
        api_token=settings.api_token,
        device=settings.device,
        clips_dir=Path(settings.clips_dir),
        classification=settings.classification,
        aqa=settings.aqa,
        demo_clips=DemoClipsConfig(
            download_enabled=download_enabled,
            force_download=force_download,
        ),
    )
