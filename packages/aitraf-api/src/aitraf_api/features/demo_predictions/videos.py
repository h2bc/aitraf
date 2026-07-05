"""S3 video URL presigning for demo prediction responses."""

from __future__ import annotations

from typing import Protocol
from urllib.parse import urlparse

import boto3
from botocore.config import Config

VIDEO_URL_EXPIRATION_SECONDS = 900


class VideoUrlPresigner(Protocol):
    def __call__(self, s3_path: str) -> str: ...


def create_video_url_presigner(
    *,
    aws_endpoint_url: str,
    aws_bucket: str,
) -> VideoUrlPresigner:
    s3_client = boto3.client(
        "s3",
        endpoint_url=aws_endpoint_url,
        config=Config(signature_version="s3v4"),
    )

    def presign_video_url(s3_path: str) -> str:
        return s3_client.generate_presigned_url(
            "get_object",
            Params={
                "Bucket": aws_bucket,
                "Key": urlparse(s3_path).path.removeprefix("/"),
            },
            ExpiresIn=VIDEO_URL_EXPIRATION_SECONDS,
            HttpMethod="GET",
        )

    return presign_video_url


__all__ = [
    "VIDEO_URL_EXPIRATION_SECONDS",
    "VideoUrlPresigner",
    "create_video_url_presigner",
]
