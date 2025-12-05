"""Helpers for turning a manifest row into VideoMAE-ready tensors."""

from pathlib import Path
from typing import Any, List

import torch
from torch.distributions import Normal
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

    if sampling_dist == "normal":
        indices = _normal_frame_indices(total_frames, num_frames)
    elif sampling_dist == "uniform":
        indices = torch.linspace(0, total_frames - 1, steps=num_frames).long().tolist()
    else:
        raise ValueError(
            f"Unsupported sampling_dist '{sampling_dist}'. Expected 'uniform' or 'normal'."
        )

    frames = [decoder[int(idx)] for idx in indices]

    return frames


def _normal_frame_indices(total_frames: int, num_frames: int) -> List[int]:
    """Create center-biased indices approximating a normal distribution over frames."""

    if num_frames == 1:
        return [int((total_frames - 1) / 2)]

    start = 0.5 / num_frames
    end = 1 - 0.5 / num_frames
    quantiles = torch.linspace(start, end, steps=num_frames)
    eps = torch.finfo(torch.float32).eps
    quantiles = quantiles.clamp(eps, 1 - eps)

    normal = Normal(0.0, 1.0)
    normalized = normal.icdf(quantiles)

    sigma = max(total_frames / 6, 1.0)
    center = (total_frames - 1) / 2
    indices = normalized * sigma + center
    indices = torch.clamp(indices, 0, total_frames - 1)

    return indices.round().long().tolist()


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
