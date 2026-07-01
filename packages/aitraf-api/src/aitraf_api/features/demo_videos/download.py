"""Demo clip download helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aitraf_core.storage.clips import ClipDownloadRequest, download_clips
from aitraf_core.storage.s3 import parse_s3_uri

from aitraf_api.config import Settings
from aitraf_api.features.demo_videos.filter import filter_demo_video_ids
from aitraf_api.manifests import read_jsonl_manifest


def build_demo_clip_download_requests(
    *,
    classification_rows: list[Mapping[str, Any]],
    aqa_rows: list[Mapping[str, Any]],
) -> list[ClipDownloadRequest]:
    demo_video_ids = filter_demo_video_ids(
        classification_rows=classification_rows,
        aqa_rows=aqa_rows,
    )
    if not demo_video_ids:
        raise RuntimeError("Current classification and AQA manifests have no matching videos.")

    classification_by_id = _rows_by_video_id(classification_rows)
    aqa_by_id = _rows_by_video_id(aqa_rows)

    return [
        ClipDownloadRequest(
            video_id=video_id,
            source_uri=_source_uri_for_video_id(
                video_id=video_id,
                classification_row=classification_by_id[video_id],
                aqa_row=aqa_by_id[video_id],
            ),
        )
        for video_id in demo_video_ids
    ]


def hydrate_demo_clips(settings: Settings) -> None:
    if not settings.demo_clips.download_enabled:
        return

    classification_rows = read_jsonl_manifest(settings.classification.manifest_path)
    aqa_rows = read_jsonl_manifest(settings.aqa.manifest_path)
    requests = build_demo_clip_download_requests(
        classification_rows=classification_rows,
        aqa_rows=aqa_rows,
    )
    download_clips(
        requests,
        clips_dir=settings.clips_dir,
        force=settings.demo_clips.force_download,
    )


def _rows_by_video_id(rows: list[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        result.setdefault(str(row["video_id"]), row)
    return result


def _source_uri_for_video_id(
    *,
    video_id: str,
    classification_row: Mapping[str, Any],
    aqa_row: Mapping[str, Any],
) -> str:
    source = classification_row.get("s3_path") or aqa_row.get("s3_path")
    if source is None or not str(source).strip():
        raise RuntimeError(f"Selected demo clip has no s3_path: {video_id}")
    source_uri = str(source)
    parse_s3_uri(source_uri)
    return source_uri


__all__ = [
    "build_demo_clip_download_requests",
    "hydrate_demo_clips",
]
