"""Shared S3 primitives."""

from aitraf_core.storage.s3 import (
    S3Settings,
    build_s3_client,
    download_s3_uri,
    iter_keys,
    load_s3_settings,
    object_exists,
    parse_s3_uri,
    presign_s3_uri,
    read_s3_uri,
    upload_s3_uri,
)

__all__ = [
    "S3Settings",
    "build_s3_client",
    "download_s3_uri",
    "iter_keys",
    "load_s3_settings",
    "object_exists",
    "parse_s3_uri",
    "presign_s3_uri",
    "read_s3_uri",
    "upload_s3_uri",
]
