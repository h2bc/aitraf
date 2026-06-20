"""Cache contract data for reusable pre-processing artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class SampledFrameCacheContract:
    video_id: str
    clip_path: str
    num_clips: int
    sample_frames: int
    sampling_dist: str
    source_mtime_ns: int | None = None
    source_size: int | None = None

    @classmethod
    def from_clip(
        cls,
        *,
        video_id: str,
        clip_path: Path | str,
        num_clips: int,
        sample_frames: int,
        sampling_dist: str,
    ) -> "SampledFrameCacheContract":
        path = Path(clip_path)
        stat = path.stat() if path.exists() else None
        return cls(
            video_id=video_id,
            clip_path=str(path),
            num_clips=num_clips,
            sample_frames=sample_frames,
            sampling_dist=sampling_dist,
            source_mtime_ns=stat.st_mtime_ns if stat else None,
            source_size=stat.st_size if stat else None,
        )

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VideoMaeFeatureCacheContract:
    video_id: str
    backbone: str
    processor_id: str
    model_version: str
    num_clips: int
    sample_frames: int
    sampling_dist: str
    preprocessing_version: str
    frame_contract: dict[str, Any] | None = None

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)


def load_cached_payload(path: Path | str, contract: dict[str, Any]) -> dict[str, Any] | None:
    cache_path = Path(path)
    if not cache_path.exists():
        return None
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if payload.get("contract") != contract:
        return None
    return payload


def save_cached_payload(
    path: Path | str,
    *,
    contract: dict[str, Any],
    payload: dict[str, Any],
) -> None:
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"contract": contract, **payload}, cache_path)


__all__ = [
    "load_cached_payload",
    "save_cached_payload",
    "SampledFrameCacheContract",
    "VideoMaeFeatureCacheContract",
]
