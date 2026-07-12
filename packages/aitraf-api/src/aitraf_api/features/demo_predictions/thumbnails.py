"""Extract demo-video thumbnails with FFmpeg."""

from __future__ import annotations

import subprocess
from pathlib import Path


class ThumbnailError(RuntimeError):
    """Raised when a required thumbnail cannot be prepared."""


def generate_thumbnail(video_path: Path, thumbnail_path: Path) -> None:
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-ss",
                "0.5",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-q:v",
                "3",
                "-y",
                str(thumbnail_path),
            ],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ThumbnailError(f"Failed to extract thumbnail from {video_path}") from exc

    if not thumbnail_path.is_file() or thumbnail_path.stat().st_size == 0:
        raise ThumbnailError(f"FFmpeg did not create thumbnail: {thumbnail_path}")


__all__ = [
    "ThumbnailError",
    "generate_thumbnail",
]
