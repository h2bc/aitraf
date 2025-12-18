"""Small helpers shared across data scripts."""

from pathlib import Path


def strip_clips_prefix(path: Path) -> Path:
    """Drop a leading 'clips/' prefix so downloads land under data/clips."""
    parts = path.parts
    if parts and parts[0] == "clips":
        parts = parts[1:]
    if not parts:
        return Path(path.name)
    return Path(*parts)
