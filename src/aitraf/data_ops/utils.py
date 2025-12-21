"""Small helpers shared across data scripts."""

from pathlib import Path
from typing import Callable, Mapping
import pandas as pd


def strip_clips_prefix(path: Path) -> Path:
    """Drop a leading 'clips/' prefix so downloads land under data/clips."""
    parts = path.parts
    if parts and parts[0] == "clips":
        parts = parts[1:]
    if not parts:
        return Path(path.name)
    return Path(*parts)


def apply_processors(
    df: pd.DataFrame,
    processors: Mapping[str, Callable[[object], object]],
) -> pd.DataFrame:
    for col, fn in processors.items():
        if col in df.columns:
            df[col] = df[col].apply(fn)
    return df


def apply_dtypes(
    df: pd.DataFrame,
    dtypes: Mapping[str, str],
) -> pd.DataFrame:
    for col, dtype in dtypes.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)
    return df
