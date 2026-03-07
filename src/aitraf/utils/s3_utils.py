"""Shared S3 helpers used by data and labeling operations."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError


@dataclass(frozen=True)
class S3Settings:
    endpoint_url: str
    region_name: str
    access_key: str
    secret_key: str
    bucket: str | None = None


def load_s3_settings(*, require_bucket: bool) -> S3Settings:
    settings = {
        "AWS_ENDPOINT_URL": os.getenv("AWS_ENDPOINT_URL"),
        "AWS_DEFAULT_REGION": os.getenv("AWS_DEFAULT_REGION"),
        "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
        "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "AWS_BUCKET": os.getenv("AWS_BUCKET"),
    }

    required = [
        "AWS_ENDPOINT_URL",
        "AWS_DEFAULT_REGION",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
    ]
    if require_bucket:
        required.append("AWS_BUCKET")

    missing = [name for name in required if not settings.get(name)]
    if missing:
        raise RuntimeError(
            "The following AWS environment variables must be set: " + ", ".join(missing)
        )

    return S3Settings(
        endpoint_url=str(settings["AWS_ENDPOINT_URL"]),
        region_name=str(settings["AWS_DEFAULT_REGION"]),
        access_key=str(settings["AWS_ACCESS_KEY_ID"]),
        secret_key=str(settings["AWS_SECRET_ACCESS_KEY"]),
        bucket=settings["AWS_BUCKET"],
    )


def build_s3_client(settings: S3Settings):
    return boto3.client(
        "s3",
        endpoint_url=settings.endpoint_url,
        region_name=settings.region_name,
        aws_access_key_id=settings.access_key,
        aws_secret_access_key=settings.secret_key,
    )


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Unsupported clip URI: {uri}")

    key = parsed.path.lstrip("/")
    if not key:
        raise ValueError(f"Invalid S3 key in URI: {uri}")

    return parsed.netloc, key


def iter_keys(s3_client, *, bucket: str, prefix: str) -> Iterator[str]:
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj.get("Key")
            if key and not key.endswith("/"):
                yield key


def object_exists(s3_client, *, bucket: str, key: str) -> bool:
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise
    else:
        return True


__all__ = [
    "S3Settings",
    "build_s3_client",
    "iter_keys",
    "load_s3_settings",
    "object_exists",
    "parse_s3_uri",
]
