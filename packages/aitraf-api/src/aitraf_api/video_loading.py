"""Shared API video loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import HTTPException

from aitraf_api.manifests import find_manifest_row_by_video_id


def load_video_row(
    *,
    manifest_path: Path,
    clips_dir: Path,
    video_id: str,
) -> dict[str, Any]:
    
    row = find_manifest_row_by_video_id(
        manifest_path=manifest_path,
        video_id=video_id,
    )

    clip_path = clips_dir / video_id
    
    if not clip_path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Video file is unavailable: {clip_path}",
        )
    return row


__all__ = ["load_video_row"]
