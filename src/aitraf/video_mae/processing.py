"""Helpers for turning a manifest row into VideoMAE-ready tensors."""

from pathlib import Path
from typing import Any, List

import torch
from torchcodec.decoders import VideoDecoder
from transformers import VideoMAEImageProcessor
from aitraf.utils import get_video_rotation_deg

from aitraf.data import schema
import kornia


def process_clip(
    manifest_row: dict[str, Any],
    processor: VideoMAEImageProcessor,
    local_clips_dir: str | Path,
    label2id: dict[str, int],
    num_frames: int,
    sampling_dist: str = "uniform",
) -> dict[str, torch.tensor]:
    """Load a local clip referenced by a manifest row and prepare VideoMAE inputs."""

    clip_path = local_clips_dir / manifest_row["video_id"]
    decoder = VideoDecoder(str(clip_path), dimension_order="NHWC")
    frames = _sample_frames(decoder, num_frames, clip_path, sampling_dist)
    frames = _rotate_frames(frames, get_video_rotation_deg(clip_path))
    processed_ts = processor(frames, return_tensors="pt")
    label = manifest_row[schema.TARGET_COLUMN]

    return {
        "pixel_values": processed_ts["pixel_values"][0],
        "labels": torch.tensor(label2id[label]),
    }


def _sample_frames(
    decoder: VideoDecoder, num_frames: int, clip_path: Path, sampling_dist: str
) -> List[torch.Tensor]:
    total_frames = len(decoder)

    if total_frames < num_frames:
        raise ValueError(
            f"Clip {clip_path} has only {total_frames} frames (<{num_frames})."
        )

    if sampling_dist == "gaussian_stochastic":
        indices = _center_weighted_indices(total_frames, num_frames)
    elif sampling_dist == "uniform":
        indices = torch.linspace(0, total_frames - 1, steps=num_frames).long().tolist()
    else:
        raise ValueError(
            f"Unsupported sampling_dist '{sampling_dist}'. Expected 'uniform' or 'gaussian_stochastic'."
        )

    frames = [decoder[int(idx)] for idx in indices]

    return frames


def _center_weighted_indices(total_frames: int, num_frames: int) -> List[int]:
    """Sample indices with higher probability near the clip center."""

    if num_frames == 1:
        return [int((total_frames - 1) / 2)]

    positions = torch.linspace(-1, 1, steps=total_frames, dtype=torch.float32)
    weights = torch.exp(-4 * positions.square())
    probs = weights / weights.sum()
    indices = torch.multinomial(probs, num_frames, replacement=False)
    indices, _ = torch.sort(indices)

    return indices.long().tolist()


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
