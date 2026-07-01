"""Demo-video service."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from aitraf_api.config import Settings
from aitraf_api.features.demo_videos.filter import filter_demo_video_ids
from aitraf_api.manifests import read_jsonl_manifest
from aitraf_api.schemas import DemoVideo


def list_demo_videos(settings: Settings) -> list[DemoVideo]:
    classification_rows = read_jsonl_manifest(settings.classification.manifest_path)
    aqa_rows = read_jsonl_manifest(settings.aqa.manifest_path)
    demo_video_ids = filter_demo_video_ids(
        classification_rows=classification_rows,
        aqa_rows=aqa_rows,
    )

    classification_rows_by_id: dict[str, dict[str, Any]] = {}
    for row in classification_rows:
        video_id = str(row["video_id"])
        classification_rows_by_id.setdefault(video_id, row)

    videos = [
        demo_video_from_row(classification_rows_by_id[video_id])
        for video_id in demo_video_ids
    ]
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
