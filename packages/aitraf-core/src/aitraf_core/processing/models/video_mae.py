"""Helpers for turning samples into VideoMAE-ready tensors."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
from transformers import VideoMAEImageProcessor

from aitraf_core.processing.video import load_sampled_video_frames


def process_sample(
    manifest_row: dict[str, Any],
    processor: VideoMAEImageProcessor,
    local_clips_dir: str | Path,
    num_frames: int,
    sampling_dist: str,
    label_key: str,
    label_transform: Callable[[Any], Any] = lambda x: x,
) -> dict[str, torch.Tensor]:
    """Load a single-sample task row and prepare VideoMAE inputs."""

    return {
        "pixel_values": process_video(
            video_id=manifest_row["video_id"],
            processor=processor,
            local_clips_dir=local_clips_dir,
            num_frames=num_frames,
            sampling_dist=sampling_dist,
        ),
        "labels": torch.as_tensor(label_transform(manifest_row[label_key])),
    }


def process_video(
    *,
    video_id: str,
    processor: VideoMAEImageProcessor,
    local_clips_dir: str | Path,
    num_frames: int,
    sampling_dist: str,
) -> torch.Tensor:
    frames, _ = load_sampled_video_frames(
        video_id=video_id,
        local_clips_dir=local_clips_dir,
        num_frames=num_frames,
        sampling_dist=sampling_dist,
    )
    processed_ts = processor(frames, return_tensors="pt")
    return processed_ts["pixel_values"][0]


__all__ = [
    "process_video",
    "process_sample",
]
