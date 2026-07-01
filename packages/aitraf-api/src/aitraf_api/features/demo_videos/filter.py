"""Demo-video filtering helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def filter_demo_video_ids(
    *,
    classification_rows: list[Mapping[str, Any]],
    aqa_rows: list[Mapping[str, Any]],
) -> list[str]:
    aqa_ids = {str(row["video_id"]) for row in aqa_rows}

    video_ids: list[str] = []
    seen: set[str] = set()
    for row in classification_rows:
        video_id = str(row["video_id"])
        if video_id not in aqa_ids or video_id in seen:
            continue
        video_ids.append(video_id)
        seen.add(video_id)
    return video_ids


__all__ = ["filter_demo_video_ids"]
