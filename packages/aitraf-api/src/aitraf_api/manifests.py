"""Shared manifest file helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import HTTPException

from aitraf_core.utils import read_jsonl_records


def read_jsonl_manifest(path: Path) -> list[dict[str, Any]]:
    try:
        rows = read_jsonl_records(path)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Required manifest does not exist: {path}",
        ) from exc
    except IsADirectoryError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Required manifest is not a file: {path}",
        ) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Manifest cannot be read: {path}",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Manifest has invalid rows: {path}: {exc}",
        ) from exc

    if not rows:
        raise HTTPException(
            status_code=503,
            detail=f"Required manifest is empty: {path}",
        )
    missing_video_id = [idx for idx, row in enumerate(rows) if not row.get("video_id")]
    if missing_video_id:
        raise HTTPException(
            status_code=503,
            detail=(
                "Manifest row is missing required video_id at index "
                f"{missing_video_id[0]}: {path}"
            ),
        )
    return rows


def find_manifest_row_by_video_id(
    *,
    manifest_path: Path,
    video_id: str,
) -> dict[str, Any]:
    rows = read_jsonl_manifest(manifest_path)
    for row in rows:
        if str(row["video_id"]) == video_id:
            return row
    raise HTTPException(
        status_code=404,
        detail=f"Selected video id is not in the current manifest: {video_id}",
    )


__all__ = [
    "find_manifest_row_by_video_id",
    "read_jsonl_manifest",
]
