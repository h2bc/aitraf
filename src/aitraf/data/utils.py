"""Small helpers shared across data scripts."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from aitraf.data import schema


def strip_clips_prefix(path: Path) -> Path:
    """Drop a leading 'clips/' prefix so downloads land under data/clips."""
    parts = path.parts
    if parts and parts[0] == "clips":
        parts = parts[1:]
    if not parts:
        return Path(path.name)
    return Path(*parts)


def resolve_clip_path(value: str, clips_dir: Path) -> Path:
    """Map a source value (s3 URI, relative path, absolute path) to local clip path."""
    if value.startswith("s3://"):
        parsed = urlparse(value)
        rel = Path(parsed.path.lstrip("/"))
        rel = strip_clips_prefix(rel)
        return clips_dir / rel
    path = Path(value)
    if path.is_absolute():
        return path
    return clips_dir / path


def video_paths_from_labels(labels_path: Path, filter_prefix: str | None = None) -> list[str]:
    """Return unique video path references from the labels export."""
    df = pd.read_json(labels_path, lines=True)
    if schema.VIDEO_COLUMN not in df.columns:
        return []
    series = df[schema.VIDEO_COLUMN].dropna().astype(str)
    if filter_prefix is not None:
        series = series[series.str.startswith(filter_prefix)]
    return series.drop_duplicates().tolist()


__all__ = ["strip_clips_prefix", "resolve_clip_path", "video_paths_from_labels"]
