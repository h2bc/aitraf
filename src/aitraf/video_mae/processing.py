"""Helpers for turning a manifest row into VideoMAE-ready tensors."""

from pathlib import Path
from typing import Any

import torch
from torchcodec.decoders import VideoDecoder
from transformers import VideoMAEImageProcessor

from aitraf.data import schema


def process_clip(
    manifest_row: dict[str, Any],
    processor: VideoMAEImageProcessor,
    local_clips_dir: str | Path,
    label2id: dict[str, int],
    num_frames: int,
) -> dict[str, torch.tensor]:
    """Load a local clip referenced by a manifest row and prepare VideoMAE inputs."""

    clip_path = local_clips_dir / manifest_row["video_id"]
    decoder = _load_decoder(clip_path)
    frames = _sample_frames(decoder, num_frames, clip_path)
    processed = processor(frames, return_tensors="pt")
    label = manifest_row[schema.TARGET_COLUMN]

    return {
        "pixel_values": processed["pixel_values"][0],
        "labels": torch.tensor(label2id[label]),
    }


def _load_decoder(clip_path: Path) -> VideoDecoder:
    return VideoDecoder(str(clip_path), dimension_order="NHWC")


def _sample_frames(decoder: VideoDecoder, num_frames: int, clip_path: Path) -> list:
    total_frames = len(decoder)

    if total_frames < num_frames:
        raise ValueError(
            f"Clip {clip_path} has only {total_frames} frames (<{num_frames})."
        )

    indices = torch.linspace(0, total_frames - 1, steps=num_frames).long().tolist()
    frames = [decoder[int(idx)].numpy() for idx in indices]

    return frames
