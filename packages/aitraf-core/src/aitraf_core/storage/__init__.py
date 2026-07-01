"""Shared storage helpers."""

from aitraf_core.storage.clips import (
    ClipDownloadRequest,
    ClipDownloadResult,
    download_clip,
    download_clips,
)
from aitraf_core.storage.s3 import (
    S3Settings,
    build_s3_client,
    iter_keys,
    load_s3_settings,
    object_exists,
    parse_s3_uri,
    presign_s3_uri,
)

__all__ = [
    "ClipDownloadRequest",
    "ClipDownloadResult",
    "S3Settings",
    "build_s3_client",
    "download_clip",
    "download_clips",
    "iter_keys",
    "load_s3_settings",
    "object_exists",
    "parse_s3_uri",
    "presign_s3_uri",
]
