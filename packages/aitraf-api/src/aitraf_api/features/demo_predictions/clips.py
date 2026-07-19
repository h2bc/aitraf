"""Publish the selected demo media as pose-rendered public objects."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

from aitraf_core.storage.s3 import (
    build_s3_client,
    download_s3_uri,
    load_s3_settings,
    object_exists,
    parse_s3_uri,
)

from aitraf_core.render import load_pose_artifact, render_pose_video
from aitraf_api.features.demo_predictions.thumbnails import generate_thumbnail

logger = logging.getLogger(__name__)

POSE_PREFIX = "poses"


class DemoClipError(RuntimeError):
    """Raised when a selected demo asset cannot be published safely."""


def build_public_url(endpoint_url: str, bucket: str, key: str) -> str:
    return f"{endpoint_url.rstrip('/')}/{bucket}/{quote(key, safe='/')}"


def derive_pose_key(video_id: str) -> str:
    """Return the private-bucket key holding this clip's pose artifact."""
    return f"{POSE_PREFIX}/{PurePosixPath(video_id).stem}.npz"


def prepare_public_demo_clips(
    rows: list[dict[str, Any]],
    *,
    aws_endpoint_url: str,
    source_bucket: str,
    public_bucket: str,
) -> list[dict[str, Any]]:
    client = _build_client(aws_endpoint_url, source_bucket)
    urls: dict[str, tuple[str, str]] = {}
    rendered = generated = reused = 0

    for row in {row["video_id"]: row for row in rows}.values():
        video_id, source_uri, source_key = _clip_values(row, source_bucket)
        video_key = f"videos/{video_id}"
        thumbnail_key = f"thumbnails/{PurePosixPath(video_id).stem}.jpg"

        video_created, thumbnail_created = _ensure_public_assets(
            client,
            video_id=video_id,
            source_bucket=source_bucket,
            source_uri=source_uri,
            source_key=source_key,
            public_bucket=public_bucket,
            video_key=video_key,
            thumbnail_key=thumbnail_key,
        )
        rendered += video_created
        generated += thumbnail_created
        reused += 2 - video_created - thumbnail_created
        urls[video_id] = (
            build_public_url(aws_endpoint_url, public_bucket, video_key),
            build_public_url(aws_endpoint_url, public_bucket, thumbnail_key),
        )

    logger.info(
        "Prepared %d public demo assets (%d videos rendered, %d thumbnails generated, %d objects reused)",
        len(urls),
        rendered,
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


def _ensure_public_assets(
    client: Any,
    *,
    video_id: str,
    source_bucket: str,
    source_uri: str,
    source_key: str,
    public_bucket: str,
    video_key: str,
    thumbnail_key: str,
) -> tuple[bool, bool]:
    """Publish the rendered clip and its thumbnail, reusing whatever exists."""
    video_exists = object_exists(client, bucket=public_bucket, key=video_key)
    thumbnail_exists = object_exists(client, bucket=public_bucket, key=thumbnail_key)
    if video_exists and thumbnail_exists:
        return False, False

    with tempfile.TemporaryDirectory(prefix="aitraf-demo-asset-") as temp_dir:
        directory = Path(temp_dir)
        try:
            if video_exists:
                rendered_path = _download_public_video(
                    client,
                    video_id=video_id,
                    public_bucket=public_bucket,
                    video_key=video_key,
                    directory=directory,
                )
            else:
                rendered_path = _render_clip(
                    client,
                    video_id=video_id,
                    source_bucket=source_bucket,
                    source_uri=source_uri,
                    source_key=source_key,
                    directory=directory,
                )
                _upload_file(
                    client,
                    path=rendered_path,
                    bucket=public_bucket,
                    key=video_key,
                    content_type="video/mp4",
                )

            if not thumbnail_exists:
                thumbnail_path = directory / "thumbnail.jpg"
                generate_thumbnail(rendered_path, thumbnail_path)
                _upload_file(
                    client,
                    path=thumbnail_path,
                    bucket=public_bucket,
                    key=thumbnail_key,
                    content_type="image/jpeg",
                )
        except DemoClipError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise DemoClipError(
                f"Failed to publish demo assets for {video_id!r}"
            ) from exc

    return not video_exists, not thumbnail_exists


def _render_clip(
    client: Any,
    *,
    video_id: str,
    source_bucket: str,
    source_uri: str,
    source_key: str,
    directory: Path,
) -> Path:
    """Download the clip and its poses, then render the overlay locally."""
    if not object_exists(client, bucket=source_bucket, key=source_key):
        raise DemoClipError(f"Missing source video: {source_uri}")

    pose_key = derive_pose_key(video_id)
    if not object_exists(client, bucket=source_bucket, key=pose_key):
        raise DemoClipError(
            f"Missing pose artifact for {video_id!r}: s3://{source_bucket}/{pose_key}"
        )

    source_path = directory / video_id
    pose_path = directory / f"{PurePosixPath(video_id).stem}.npz"
    rendered_path = directory / f"rendered-{video_id}"

    download_s3_uri(source_uri, source_path, s3_client=client)
    download_s3_uri(f"s3://{source_bucket}/{pose_key}", pose_path, s3_client=client)
    render_pose_video(source_path, load_pose_artifact(pose_path), rendered_path)
    return rendered_path


def _download_public_video(
    client: Any,
    *,
    video_id: str,
    public_bucket: str,
    video_key: str,
    directory: Path,
) -> Path:
    """Fetch the already-published rendered clip instead of rendering again."""
    rendered_path = directory / f"rendered-{video_id}"
    download_s3_uri(
        f"s3://{public_bucket}/{video_key}", rendered_path, s3_client=client
    )
    return rendered_path


def _upload_file(
    client: Any,
    *,
    path: Path,
    bucket: str,
    key: str,
    content_type: str,
) -> None:
    with path.open("rb") as body:
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ContentType=content_type,
        )


__all__ = [
    "DemoClipError",
    "build_public_url",
    "derive_pose_key",
    "prepare_public_demo_clips",
]
