"""Publish the selected demo media to stable public objects."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

from aitraf_core.storage.s3 import (
    build_s3_client,
    copy_s3_object,
    download_s3_uri,
    load_s3_settings,
    object_exists,
    parse_s3_uri,
)

from aitraf_api.features.demo_predictions.thumbnails import generate_thumbnail

logger = logging.getLogger(__name__)


class DemoClipError(RuntimeError):
    """Raised when a selected demo asset cannot be published safely."""


def build_public_url(endpoint_url: str, bucket: str, key: str) -> str:
    return f"{endpoint_url.rstrip('/')}/{bucket}/{quote(key, safe='/')}"


def prepare_public_demo_clips(
    rows: list[dict[str, Any]],
    *,
    aws_endpoint_url: str,
    source_bucket: str,
    public_bucket: str,
) -> list[dict[str, Any]]:
    client = _build_client(aws_endpoint_url, source_bucket)
    urls: dict[str, tuple[str, str]] = {}
    copied = generated = reused = 0

    for row in {row["video_id"]: row for row in rows}.values():
        video_id, source_uri, source_key = _clip_values(row, source_bucket)
        video_key = f"videos/{video_id}"
        thumbnail_key = f"thumbnails/{PurePosixPath(video_id).stem}.jpg"
        video_created = _ensure_public_video(
            client, source_bucket, source_uri, source_key, public_bucket, video_key
        )
        thumbnail_created = _ensure_public_thumbnail(
            client, video_id, source_uri, public_bucket, thumbnail_key
        )
        copied += video_created
        generated += thumbnail_created
        reused += 2 - video_created - thumbnail_created
        urls[video_id] = (
            build_public_url(aws_endpoint_url, public_bucket, video_key),
            build_public_url(aws_endpoint_url, public_bucket, thumbnail_key),
        )

    logger.info(
        "Prepared %d public demo assets (%d videos copied, %d thumbnails generated, %d objects reused)",
        len(urls),
        copied,
        generated,
        reused,
    )
    prepared_rows = []
    for row in rows:
        video_url, thumbnail_url = urls[row["video_id"]]
        prepared_rows.append(
            {**row, "video_url": video_url, "thumbnail_url": thumbnail_url}
        )
    return prepared_rows


def _build_client(aws_endpoint_url: str, source_bucket: str) -> Any:
    settings = load_s3_settings(require_bucket=True)
    if settings.endpoint_url != aws_endpoint_url or settings.bucket != source_bucket:
        raise DemoClipError("Configured S3 settings do not match API settings")
    return build_s3_client(settings)


def _clip_values(row: dict[str, Any], source_bucket: str) -> tuple[str, str, str]:
    video_id = row["video_id"]
    name = PurePosixPath(video_id)
    if not video_id or name.name != video_id or not name.suffix:
        raise DemoClipError(f"Invalid video_id: {video_id!r}")
    source_uri = row["s3_path"]
    row_bucket, source_key = parse_s3_uri(source_uri)
    if row_bucket != source_bucket:
        raise DemoClipError(f"Expected source bucket {source_bucket!r}: {source_uri!r}")
    return video_id, source_uri, source_key


def _ensure_public_video(
    client: Any, source_bucket: str, source_uri: str, source_key: str,
    public_bucket: str, video_key: str,
) -> bool:
    if not object_exists(client, bucket=source_bucket, key=source_key):
        raise DemoClipError(f"Missing source video: {source_uri}")
    if object_exists(client, bucket=public_bucket, key=video_key):
        return False
    copy_s3_object(
        client,
        source_bucket=source_bucket,
        source_key=source_key,
        destination_bucket=public_bucket,
        destination_key=video_key,
    )
    return True


def _ensure_public_thumbnail(
    client: Any, video_id: str, source_uri: str,
    public_bucket: str, thumbnail_key: str,
) -> bool:
    if object_exists(client, bucket=public_bucket, key=thumbnail_key):
        return False
    with tempfile.TemporaryDirectory(prefix="aitraf-thumbnail-") as temp_dir:
        try:
            thumbnail_path = _create_thumbnail(
                client,
                video_id=video_id,
                source_uri=source_uri,
                directory=Path(temp_dir),
            )
            _upload_thumbnail(
                client,
                thumbnail_path=thumbnail_path,
                public_bucket=public_bucket,
                thumbnail_key=thumbnail_key,
            )
        except Exception as exc:  # noqa: BLE001
            raise DemoClipError(
                f"Failed to publish thumbnail for {video_id!r}"
            ) from exc
    return True


def _create_thumbnail(
    client: Any, *, video_id: str, source_uri: str, directory: Path
) -> Path:
    video_path = directory / video_id
    thumbnail_path = directory / "thumbnail.jpg"
    download_s3_uri(source_uri, video_path, s3_client=client)
    generate_thumbnail(video_path, thumbnail_path)
    return thumbnail_path


def _upload_thumbnail(
    client: Any,
    *,
    thumbnail_path: Path,
    public_bucket: str,
    thumbnail_key: str,
) -> None:
    with thumbnail_path.open("rb") as body:
        client.put_object(
            Bucket=public_bucket,
            Key=thumbnail_key,
            Body=body,
            ContentType="image/jpeg",
        )


__all__ = [
    "DemoClipError",
    "build_public_url",
    "prepare_public_demo_clips",
]
