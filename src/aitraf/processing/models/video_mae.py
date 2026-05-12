"""Helpers for turning samples into VideoMAE-ready tensors."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, List

import kornia
import torch
from torchcodec.decoders import VideoDecoder
from transformers import VideoMAEImageProcessor

from aitraf.processing.utils import sample_frame_indices
from aitraf.utils.video_utils import get_video_rotation_deg


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
        "pixel_values": _process_video(
            video_id=manifest_row["video_id"],
            processor=processor,
            local_clips_dir=local_clips_dir,
            num_frames=num_frames,
            sampling_dist=sampling_dist,
        ),
        "labels": torch.as_tensor(label_transform(manifest_row[label_key])),
    }


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
        "left_pixel_values": _process_video(
            video_id=sample["left_video_id"],
            processor=processor,
            local_clips_dir=local_clips_dir,
            num_frames=num_frames,
            sampling_dist=sampling_dist,
        ),
        "right_pixel_values": _process_video(
            video_id=sample["right_video_id"],
            processor=processor,
            local_clips_dir=local_clips_dir,
            num_frames=num_frames,
            sampling_dist=sampling_dist,
        ),
        "labels": torch.as_tensor(label_transform(sample["pair_label"])),
    }


def _process_video(
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


def load_sampled_video_frames(
    *,
    video_id: str,
    local_clips_dir: str | Path,
    num_frames: int,
    sampling_dist: str,
) -> tuple[List[torch.Tensor], list[int]]:
    clip_path = Path(local_clips_dir) / video_id
    decoder = VideoDecoder(str(clip_path), dimension_order="NHWC")
    frame_indices = sample_frame_indices(
        len(decoder),
        num_frames,
        sampling_dist,
        source=clip_path,
    )
    frames = [decoder[int(idx)] for idx in frame_indices]
    frames = _rotate_frames(frames, get_video_rotation_deg(clip_path))
    return frames, frame_indices


def _rotate_frames(frames: List[torch.Tensor], rotation_deg: int) -> List[torch.Tensor]:
    if rotation_deg == 0:
        return frames

    x = torch.stack(frames)  # B, (H, W, C) -> (B, H, W, C)
    x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
    x = x.float()  # to float

    angles = torch.full(
        (len(frames),), float(rotation_deg), dtype=torch.float32, device=x.device
    )
    x = kornia.geometry.transform.rotate(x, angles)

    x = x.clamp(0, 255).byte()  # to uint8
    return list(x.permute(0, 2, 3, 1))  # (B, C, H, W) -> B, (H, W, C)


__all__ = [
    "load_sampled_video_frames",
    "process_pair_sample",
    "process_sample",
]
