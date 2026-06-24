"""File cache helpers."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")


def with_file_cache(
    *,
    path: Path | str,
    compute: Callable[[], T],
    force: bool = False,
    load: Callable[[Path], T] | None = None,
    save: Callable[[Path, T], object] | None = None,
    on_cache_hit: Callable[[Path], None] | None = None,
    on_cache_write: Callable[[Path], None] | None = None,
) -> T | None:
    cache_path = Path(path)
    if cache_path.exists() and not force:
        value = load(cache_path) if load is not None else None
        if on_cache_hit is not None:
            on_cache_hit(cache_path)
        return value

    value = compute()
    if save is not None:
        save(cache_path, value)
    if on_cache_write is not None:
        on_cache_write(cache_path)
    return value


__all__ = [
    "with_file_cache",
]
