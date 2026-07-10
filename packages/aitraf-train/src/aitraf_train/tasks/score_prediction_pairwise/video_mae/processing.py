"""VideoMAE processing helpers for pairwise score prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
from transformers import VideoMAEImageProcessor

from aitraf_ml_core.processing.models.video_mae import process_video


def process_pair_sample(
    sample: dict[str, Any],
    processor: VideoMAEImageProcessor,
    local_clips_dir: str | Path,
    num_frames: int,
    sampling_dist: str,
    label_transform: Callable[[Any], Any] = lambda x: x,
) -> dict[str, torch.Tensor]:
    """Load a left/right clip pair and prepare VideoMAE inputs."""

    return {
        "left_pixel_values": process_video(
            video_id=sample["left_video_id"],
            processor=processor,
            local_clips_dir=local_clips_dir,
            num_frames=num_frames,
            sampling_dist=sampling_dist,
        ),
        "right_pixel_values": process_video(
            video_id=sample["right_video_id"],
            processor=processor,
            local_clips_dir=local_clips_dir,
            num_frames=num_frames,
            sampling_dist=sampling_dist,
        ),
        "labels": torch.as_tensor(label_transform(sample["pair_label"])),
    }


__all__ = ["process_pair_sample"]
