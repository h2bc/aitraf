"""Train-owned storage orchestration."""

from aitraf_train.storage.clips import (
    ClipDownloadRequest,
    ClipDownloadResult,
    download_clip,
    download_clips,
)

__all__ = [
    "ClipDownloadRequest",
    "ClipDownloadResult",
    "download_clip",
    "download_clips",
]
