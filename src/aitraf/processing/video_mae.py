"""Helpers for turning a manifest row into VideoMAE-ready tensors."""

from functools import reduce
from pathlib import Path
from typing import Any, Callable, List

import torch
from torchcodec.decoders import VideoDecoder
from transformers import VideoMAEImageProcessor
from aitraf.utils import get_video_rotation_deg

from aitraf.processing import sample_frame_indices
import kornia


def process_clip(
    manifest_row: dict[str, Any],
    processor: VideoMAEImageProcessor,
    local_clips_dir: str | Path,
    label2id: dict[str, int],
    num_frames: int,
    sampling_dist: str,
    target_column: str,
) -> dict[str, torch.tensor]:
    """Load a local clip referenced by a manifest row and prepare VideoMAE inputs."""

    clip_path = local_clips_dir / manifest_row["video_id"]
    decoder = VideoDecoder(str(clip_path), dimension_order="NHWC")
    frame_indices = sample_frame_indices(
        len(decoder),
        num_frames,
        sampling_dist,
        source=clip_path,
    )
    frames = [decoder[int(idx)] for idx in frame_indices]
    frames = _rotate_frames(frames, get_video_rotation_deg(clip_path))
    processed_ts = processor(frames, return_tensors="pt")
    label = manifest_row[target_column]

    return {
        "pixel_values": processed_ts["pixel_values"][0],
        "labels": torch.tensor(label2id[label]),
    }


def build_collate(
    processor: VideoMAEImageProcessor,
    clips_dir: Path | str,
    label2id: dict[str, int],
    sample_frames: int,
    sampling_dist: str,
    target_column: str,
) -> Callable[[list[dict[str, Any]]], dict[str, torch.Tensor]]:
    """Create a collate_fn consistent across training and eval."""

    def _collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        processed_batch = [
            process_clip(
                row,
                processor,
                clips_dir,
                label2id,
                sample_frames,
                sampling_dist,
                target_column,
            )
            for row in batch
        ]

        pivot = reduce(
            lambda acc, x: {k: acc.get(k, []) + [x[k]] for k in x},
            processed_batch,
            {},
        )

        return {k: torch.stack(v) for k, v in pivot.items()}

    return _collate


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
