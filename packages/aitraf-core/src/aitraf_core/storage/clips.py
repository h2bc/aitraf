"""Shared clip download helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from aitraf_core.storage.s3 import build_s3_client, load_s3_settings, parse_s3_uri


ClipDownloadStatus = Literal["downloaded", "skipped-existing"]


@dataclass(frozen=True)
class ClipDownloadRequest:
    video_id: str
    source_uri: str
    relative_path: Path | str | None = None

    def destination_path(self, clips_dir: Path | str) -> Path:
        relative = Path(self.relative_path if self.relative_path is not None else self.video_id)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Clip destination must be relative: {relative}")
        return Path(clips_dir) / relative


@dataclass(frozen=True)
class ClipDownloadResult:
    video_id: str
    source_uri: str
    destination_path: Path
    status: ClipDownloadStatus


def download_clip(
    request: ClipDownloadRequest,
    *,
    clips_dir: Path | str,
    force: bool = False,
    s3_client=None,
) -> ClipDownloadResult:
    destination = request.destination_path(clips_dir)
    if destination.exists() and not force:
        return ClipDownloadResult(
            video_id=request.video_id,
            source_uri=request.source_uri,
            destination_path=destination,
            status="skipped-existing",
        )

    bucket, key = parse_s3_uri(request.source_uri)
    client = s3_client
    if client is None:
        client = build_s3_client(load_s3_settings(require_bucket=False))

    destination.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(bucket, key, str(destination))
    return ClipDownloadResult(
        video_id=request.video_id,
        source_uri=request.source_uri,
        destination_path=destination,
        status="downloaded",
    )


def download_clips(
    requests: list[ClipDownloadRequest],
    *,
    clips_dir: Path | str,
    force: bool = False,
    s3_client=None,
    on_result: Callable[[ClipDownloadResult, int, int], None] | None = None,
) -> list[ClipDownloadResult]:
    client = s3_client
    results: list[ClipDownloadResult] = []
    total = len(requests)
    for idx, request in enumerate(requests, start=1):
        destination = request.destination_path(clips_dir)
        if not destination.exists() or force:
            if client is None:
                client = build_s3_client(load_s3_settings(require_bucket=False))
        result = download_clip(
            request,
            clips_dir=clips_dir,
            force=force,
            s3_client=client,
        )
        results.append(result)
        if on_result is not None:
            on_result(result, idx, total)
    return results


__all__ = [
    "ClipDownloadRequest",
    "ClipDownloadResult",
    "download_clip",
    "download_clips",
]
