"""Demo-video service."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from aitraf_api.config import Settings
from aitraf_api.manifests import read_jsonl_manifest
from aitraf_api.schemas import DemoVideo


def list_demo_videos(settings: Settings) -> list[DemoVideo]:
    classification_rows = read_jsonl_manifest(settings.classification.manifest_path)
    aqa_rows = read_jsonl_manifest(settings.aqa.manifest_path)
    aqa_ids = {str(row["video_id"]) for row in aqa_rows}

    videos: list[DemoVideo] = []
    seen: set[str] = set()
    for row in classification_rows:
        video_id = str(row["video_id"])
        if video_id not in aqa_ids or video_id in seen:
            continue
        videos.append(demo_video_from_row(row))
        seen.add(video_id)

    if not videos:
        raise HTTPException(
            status_code=503,
            detail="Current classification and AQA manifests have no matching videos",
        )
    return videos


def demo_video_from_row(row: dict[str, Any]) -> DemoVideo:
    video_id = str(row["video_id"])
    return DemoVideo(
        id=video_id,
        video_id=video_id,
        s3_path=_optional_str(row.get("s3_path")),
        person=_optional_str(row.get("person")),
        trick=_optional_str(row.get("trick")),
        execution_score=row.get("execution_score"),
    )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


__all__ = [
    "demo_video_from_row",
    "list_demo_videos",
]
