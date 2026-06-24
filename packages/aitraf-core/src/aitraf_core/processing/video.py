"""Generic video loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import List

import av
import kornia
import torch
from torchcodec.decoders import VideoDecoder

from aitraf_core.processing.sampling import sample_frame_indices


def get_video_rotation_deg(path: Path) -> int:
    with av.open(str(path)) as container:
        frame = next(container.decode(video=0))
        rotation_deg = getattr(frame, "rotation", None)
        if rotation_deg is None:
            raise ValueError(f"No rotation metadata found: {path}")

        return (rotation_deg + 360) % 360


def sample_video_frames(
    *,
    decoder: VideoDecoder,
    video_path: Path,
    frame_range: tuple[int, int],
    num_frames: int,
    sampling_dist: str,
) -> List[torch.Tensor]:
    frame_indices = sample_frame_indices(
        frame_range=frame_range,
        num_frames=num_frames,
        sampling_dist=sampling_dist,
        source=video_path,
    )

    return decode_video_frames(
        decoder=decoder,
        video_path=video_path,
        frame_indices=frame_indices,
    )


def decode_video_frames(
    *,
    decoder: VideoDecoder,
    video_path: Path,
    frame_indices: list[int],
) -> List[torch.Tensor]:
    frames = list(decoder.get_frames_at(frame_indices).data)
    return rotate_frames(frames, get_video_rotation_deg(video_path))


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
    "decode_video_frames",
    "get_video_rotation_deg",
    "rotate_frames",
    "sample_video_frames",
]
