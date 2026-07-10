"""S3 media URL presigning for demo prediction responses."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol
import logging
from time import monotonic
from aitraf_core.storage.s3 import (
    build_s3_client,
    download_s3_uri,
    load_s3_settings,
    parse_s3_uri,
    presign_s3_uri,
)

VIDEO_URL_EXPIRATION_SECONDS = 900
logger = logging.getLogger("uvicorn.error")


class VideoDownloadError(RuntimeError):
    """Raised when a required demo video cannot be downloaded."""


def download_prediction_videos(
    rows: list[dict[str, Any]],
    *,
    aws_endpoint_url: str,
    aws_bucket: str,
    directory: Path,
) -> dict[str, Path]:
    settings = load_s3_settings(require_bucket=True)
    if settings.endpoint_url != aws_endpoint_url or settings.bucket != aws_bucket:
        raise VideoDownloadError("Configured S3 settings do not match API settings")
    s3_client = build_s3_client(settings)
    local_videos: dict[str, Path] = {}
    started_at = monotonic()
    logger.info("Downloading %d demo videos", len(rows))
    for index, row in enumerate(rows):
        source_path = row["s3_path"]
        source_bucket, source_key = parse_s3_uri(source_path)
        if source_bucket != aws_bucket:
            raise VideoDownloadError(
                f"Invalid clip S3 path for bucket {aws_bucket!r}: {source_path!r}"
            )
        destination = directory / f"{index}{Path(source_key).suffix}"
        try:
            download_s3_uri(source_path, destination, s3_client=s3_client)
        except Exception as exc:  # noqa: BLE001
            raise VideoDownloadError(f"Failed to download {source_path}") from exc
        local_videos[source_path] = destination
        logger.info(
            "Downloaded demo video %d/%d: %s", index + 1, len(rows), row["video_id"]
        )
    logger.info(
        "Downloaded %d demo videos in %.2fs", len(rows), monotonic() - started_at
    )
    return local_videos


class AssetUrlPresigner(Protocol):
    def __call__(self, s3_path: str) -> str: ...


def create_asset_url_presigner(
    *,
    aws_endpoint_url: str,
    aws_bucket: str,
) -> AssetUrlPresigner:
    settings = load_s3_settings(require_bucket=True)
    if settings.endpoint_url != aws_endpoint_url or settings.bucket != aws_bucket:
        raise RuntimeError("Configured S3 settings do not match API settings")
    s3_client = build_s3_client(settings)

    def presign_asset_url(s3_path: str) -> str:
        return presign_s3_uri(
            s3_path,
            expires_in=VIDEO_URL_EXPIRATION_SECONDS,
            s3_client=s3_client,
        )

    return presign_asset_url


__all__ = [
    "VIDEO_URL_EXPIRATION_SECONDS",
    "AssetUrlPresigner",
    "VideoDownloadError",
    "create_asset_url_presigner",
    "download_prediction_videos",
]
