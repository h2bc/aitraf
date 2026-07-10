"""Helpers for turning samples into VideoMAE-ready tensors."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
from transformers import VideoMAEImageProcessor
from torchcodec.decoders import VideoDecoder

from aitraf_ml_core.processing.video import sample_video_frames


def process(
    *,
    inputs: list[torch.Tensor] | list[list[torch.Tensor]],
    processor: VideoMAEImageProcessor,
) -> torch.Tensor:
    return processor(inputs, return_tensors="pt")["pixel_values"]


def process_sample(
    manifest_row: dict[str, Any],
    processor: VideoMAEImageProcessor,
    local_clips_dir: Path,
    num_frames: int,
    sampling_dist: str,
    label_key: str,
    label_transform: Callable[[Any], Any] = lambda x: x,
) -> dict[str, torch.Tensor]:
    """Load a single-sample task row and prepare VideoMAE inputs."""

    return {
        "pixel_values": process_video_mae_clip(
            video_id=manifest_row["video_id"],
            processor=processor,
            local_clips_dir=local_clips_dir,
            num_frames=num_frames,
            sampling_dist=sampling_dist,
        ),
        "labels": torch.as_tensor(label_transform(manifest_row[label_key])),
    }


def process_video_mae_clip(
    *,
    video_id: str,
    processor: VideoMAEImageProcessor,
    local_clips_dir: Path,
    num_frames: int,
    sampling_dist: str,
) -> torch.Tensor:
    video_path = local_clips_dir / video_id
    decoder = VideoDecoder(str(video_path), dimension_order="NHWC")
    frames = sample_video_frames(
        decoder=decoder,
        video_path=video_path,
        frame_range=(0, len(decoder)),
        num_frames=num_frames,
        sampling_dist=sampling_dist,
    )

    return process(inputs=frames, processor=processor)[0]


def process_video(
    *,
    video_id: str,
    processor: VideoMAEImageProcessor,
    local_clips_dir: Path,
    num_frames: int,
    sampling_dist: str,
) -> torch.Tensor:
    return process_video_mae_clip(
        video_id=video_id,
        processor=processor,
        local_clips_dir=local_clips_dir,
        num_frames=num_frames,
        sampling_dist=sampling_dist,
    )


__all__ = [
    "process",
    "process_video",
    "process_video_mae_clip",
    "process_sample",
]
