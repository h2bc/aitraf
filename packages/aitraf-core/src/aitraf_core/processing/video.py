"""Generic video loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import List

import kornia
import torch
from torchcodec.decoders import VideoDecoder

from aitraf_core.processing.utils import sample_frame_indices
from aitraf_core.utils.video_utils import get_video_rotation_deg


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
    frames = list(decoder.get_frames_at(frame_indices).data)
    frames = rotate_frames(frames, get_video_rotation_deg(clip_path))
    return frames, frame_indices


def load_segmented_video_frames(
    *,
    video_id: str,
    local_clips_dir: str | Path,
    num_segments: int,
    frames_per_segment: int,
    sampling_dist: str,
) -> tuple[list[List[torch.Tensor]], list[list[int]]]:
    clip_path = Path(local_clips_dir) / video_id
    decoder = VideoDecoder(str(clip_path), dimension_order="NHWC")
    total_frames = len(decoder)
    if total_frames <= 0:
        raise ValueError(f"Video has no frames: {clip_path}")

    boundaries = torch.linspace(0, total_frames, steps=num_segments + 1).long().tolist()

    segment_indices = [
        _sample_segment_frame_indices(
            start=boundaries[idx],
            end=boundaries[idx + 1],
            num_frames=frames_per_segment,
            sampling_dist=sampling_dist,
            source=clip_path,
            total_frames=total_frames,
        )
        for idx in range(num_segments)
    ]
    flat_indices = [frame_idx for segment in segment_indices for frame_idx in segment]
    frames = list(decoder.get_frames_at(flat_indices).data)
    frames = rotate_frames(frames, get_video_rotation_deg(clip_path))

    segmented_frames = [
        frames[idx * frames_per_segment : (idx + 1) * frames_per_segment]
        for idx in range(num_segments)
    ]
    return segmented_frames, segment_indices


def _sample_segment_frame_indices(
    *,
    start: int,
    end: int,
    num_frames: int,
    sampling_dist: str,
    source: Path | str,
    total_frames: int,
) -> list[int]:
    if end <= start:
        start = min(start, total_frames - 1)
        end = start + 1

    segment_frames = end - start
    if segment_frames >= num_frames:
        return [
            start + idx
            for idx in sample_frame_indices(
                segment_frames,
                num_frames,
                sampling_dist,
                source=source,
            )
        ]

    indices = torch.linspace(0, segment_frames - 1, steps=num_frames).long().tolist()
    return [start + idx for idx in indices]


def rotate_frames(frames: List[torch.Tensor], rotation_deg: int) -> List[torch.Tensor]:
    if rotation_deg == 0:
        return frames

    x = torch.stack(frames)  # B, (H, W, C) -> (B, H, W, C)
    x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
    x = x.float()

    angles = torch.full(
        (len(frames),), float(rotation_deg), dtype=torch.float32, device=x.device
    )
    x = kornia.geometry.transform.rotate(x, angles)

    x = x.clamp(0, 255).byte()
    return list(x.permute(0, 2, 3, 1))  # (B, C, H, W) -> B, (H, W, C)


__all__ = [
    "load_sampled_video_frames",
    "load_segmented_video_frames",
    "rotate_frames",
]
