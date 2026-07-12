"""Shared S3 helpers."""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import boto3


@dataclass(frozen=True)
class S3Settings:
    endpoint_url: str
    region_name: str
    access_key: str
    secret_key: str
    bucket: str | None = None


def load_s3_settings(*, require_bucket: bool) -> S3Settings:
    settings = {
        "AWS_ENDPOINT_URL": _read_env_value("AWS_ENDPOINT_URL"),
        "AWS_DEFAULT_REGION": _read_env_value("AWS_DEFAULT_REGION"),
        "AWS_ACCESS_KEY_ID": _read_env_value("AWS_ACCESS_KEY_ID"),
        "AWS_SECRET_ACCESS_KEY": _read_env_value("AWS_SECRET_ACCESS_KEY"),
        "AWS_BUCKET": _read_env_value("AWS_BUCKET"),
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


def _read_env_value(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


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


def presign_s3_uri(
    uri: str,
    *,
    expires_in: int = 7 * 24 * 60 * 60,
    s3_client=None,
) -> str:
    bucket, key = parse_s3_uri(uri)
    client = s3_client

    if client is None:
        client = build_s3_client(load_s3_settings(require_bucket=False))

    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires_in,
    )


def download_s3_uri(uri: str, destination: Path, *, s3_client) -> None:
    bucket, key = parse_s3_uri(uri)
    body = s3_client.get_object(Bucket=bucket, Key=key)["Body"]
    try:
        with destination.open("wb") as output:
            while chunk := body.read(1024 * 1024):
                output.write(chunk)
    finally:
        body.close()


def read_s3_uri(uri: str, *, s3_client) -> bytes:
    bucket, key = parse_s3_uri(uri)
    body = s3_client.get_object(Bucket=bucket, Key=key)["Body"]
    try:
        return body.read()
    finally:
        body.close()


def upload_s3_uri(source: Path, uri: str, *, s3_client) -> None:
    bucket, key = parse_s3_uri(uri)
    with source.open("rb") as body:
        s3_client.put_object(Bucket=bucket, Key=key, Body=body)


def iter_keys(s3_client, *, bucket: str, prefix: str) -> Iterator[str]:
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj.get("Key")
            if key and not key.endswith("/"):
                yield key


def object_exists(s3_client, *, bucket: str, key: str) -> bool:
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=key, MaxKeys=1)
    return any(item.get("Key") == key for item in response.get("Contents", []))


def copy_s3_object(
    s3_client,
    *,
    source_bucket: str,
    source_key: str,
    destination_bucket: str,
    destination_key: str,
) -> None:
    s3_client.copy_object(
        Bucket=destination_bucket,
        Key=destination_key,
        CopySource={"Bucket": source_bucket, "Key": source_key},
    )


__all__ = [
    "S3Settings",
    "build_s3_client",
    "download_s3_uri",
    "copy_s3_object",
    "iter_keys",
    "load_s3_settings",
    "object_exists",
    "parse_s3_uri",
    "presign_s3_uri",
    "read_s3_uri",
    "upload_s3_uri",
]
