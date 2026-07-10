"""Create and persist missing demo-video thumbnails in S3."""

from __future__ import annotations

import subprocess
import tempfile
import logging
from pathlib import Path, PurePosixPath
from time import monotonic
from typing import Any

from aitraf_core.storage.s3 import (
    build_s3_client,
    load_s3_settings,
    object_exists,
    parse_s3_uri,
    upload_s3_uri,
)

THUMBNAIL_PREFIX = "thumbnails"
logger = logging.getLogger("uvicorn.error")


class ThumbnailError(RuntimeError):
    """Raised when a required thumbnail cannot be prepared."""


def create_thumbnail_s3_path(video_id: str, *, bucket: str) -> str:
    video_name = PurePosixPath(video_id)
    if not video_id or video_name.name != video_id or not video_name.suffix:
        raise ThumbnailError(f"Invalid video_id: {video_id!r}")
    thumbnail_key = PurePosixPath(THUMBNAIL_PREFIX, video_name).with_suffix(".jpg")
    return f"s3://{bucket}/{thumbnail_key.as_posix()}"


def ensure_prediction_thumbnails(
    rows: list[dict[str, Any]],
    *,
    local_videos: dict[str, Path],
    aws_endpoint_url: str,
    aws_bucket: str,
) -> list[dict[str, Any]]:
    settings = load_s3_settings(require_bucket=True)
    if settings.endpoint_url != aws_endpoint_url or settings.bucket != aws_bucket:
        raise ThumbnailError("Configured S3 settings do not match API settings")
    s3_client = build_s3_client(settings)
    prepared_rows: list[dict[str, Any]] = []
    generated = 0
    started_at = monotonic()
    logger.info("Preparing thumbnails for %d demo videos", len(rows))
    for index, row in enumerate(rows):
        source_path = row["s3_path"]
        thumbnail_path = create_thumbnail_s3_path(row["video_id"], bucket=aws_bucket)
        _, thumbnail_key = parse_s3_uri(thumbnail_path)
        if not object_exists(s3_client, bucket=aws_bucket, key=thumbnail_key):
            video_path = local_videos[source_path]
            _generate_and_upload_thumbnail(
                s3_client,
                bucket=aws_bucket,
                video_path=video_path,
                thumbnail_key=thumbnail_key,
            )
            generated += 1
            logger.info(
                "Generated thumbnail %d/%d: %s", index + 1, len(rows), row["video_id"]
            )
        prepared_rows.append({**row, "thumbnail_s3_path": thumbnail_path})
    logger.info(
        "Prepared %d thumbnails (%d generated, %d reused) in %.2fs",
        len(rows),
        generated,
        len(rows) - generated,
        monotonic() - started_at,
    )
    return prepared_rows


def find_rows_with_missing_thumbnails(
    rows: list[dict[str, Any]],
    *,
    aws_endpoint_url: str,
    aws_bucket: str,
) -> list[dict[str, Any]]:
    settings = load_s3_settings(require_bucket=True)
    if settings.endpoint_url != aws_endpoint_url or settings.bucket != aws_bucket:
        raise ThumbnailError("Configured S3 settings do not match API settings")
    s3_client = build_s3_client(settings)
    return [
        row
        for row in rows
        if not object_exists(
            s3_client,
            bucket=aws_bucket,
            key=parse_s3_uri(
                create_thumbnail_s3_path(row["video_id"], bucket=aws_bucket)
            )[1],
        )
    ]


def _generate_and_upload_thumbnail(
    s3_client: Any,
    *,
    bucket: str,
    video_path: Path,
    thumbnail_key: str,
) -> None:
    with tempfile.TemporaryDirectory(prefix="aitraf-thumbnail-") as temp_dir:
        thumbnail_path = Path(temp_dir) / "thumbnail.jpg"
        try:
            generate_thumbnail(video_path, thumbnail_path)
            upload_s3_uri(
                thumbnail_path,
                f"s3://{bucket}/{thumbnail_key}",
                s3_client=s3_client,
            )
        except ThumbnailError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ThumbnailError(
                f"Failed to create s3://{bucket}/{thumbnail_key} from {video_path}"
            ) from exc


def generate_thumbnail(video_path: Path, thumbnail_path: Path) -> None:
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-ss",
                "0.5",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-q:v",
                "3",
                "-y",
                str(thumbnail_path),
            ],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ThumbnailError(f"Failed to extract thumbnail from {video_path}") from exc

    if not thumbnail_path.is_file() or thumbnail_path.stat().st_size == 0:
        raise ThumbnailError(f"FFmpeg did not create thumbnail: {thumbnail_path}")


__all__ = [
    "ThumbnailError",
    "create_thumbnail_s3_path",
    "ensure_prediction_thumbnails",
    "find_rows_with_missing_thumbnails",
    "generate_thumbnail",
]
