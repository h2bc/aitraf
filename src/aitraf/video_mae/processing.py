"""Helpers for turning a manifest row into VideoMAE-ready tensors."""

from pathlib import Path
from typing import Any

import torch
from torchcodec.decoders import VideoDecoder
from transformers import VideoMAEImageProcessor

from aitraf.data import schema


def load_clip(
    *,
    row: dict[str, Any],
    processor: VideoMAEImageProcessor,
    clips_dir: str | Path,
    num_frames: int = 16,
    device: str = "cuda",
    label2id: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Load a local clip referenced by a manifest row and prepare VideoMAE inputs."""

    clip_path = _resolve_clip(row=row, clips_dir=clips_dir)
    decoder = _load_decoder(clip_path, device)
    frames = _sample_frames(decoder, num_frames, clip_path)
    processed = processor(frames, return_tensors="pt")
    label = _resolve_label_id(row, label2id)
    metadata = {k: v for k, v in row.items() if k != schema.TARGET_COLUMN}

    return {
        "pixel_values": processed["pixel_values"][0],
        "label": label,
        "metadata": metadata,
        "clip_path": clip_path,
    }


def _resolve_clip(*, row: dict[str, Any], clips_dir: str | Path) -> Path:
    clip_id = row.get("video_id")

    if not clip_id:
        raise KeyError("Manifest row missing 'video_id'.")

    clip_path = Path(clips_dir) / clip_id

    if not clip_path.exists():
        raise FileNotFoundError(f"Clip not found: {clip_path}")
    return clip_path


def _load_decoder(clip_path: Path, device: str) -> VideoDecoder:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available for video decoding")

    return VideoDecoder(str(clip_path), dimension_order="NHWC", device=device)


def _sample_frames(decoder: VideoDecoder, num_frames: int, clip_path: Path) -> list:
    total_frames = len(decoder)

    if total_frames < num_frames:
        raise ValueError(
            f"Clip {clip_path} has only {total_frames} frames (<{num_frames})."
        )

    indices = torch.linspace(0, total_frames - 1, steps=num_frames).long().tolist()
    frames = [decoder[int(idx)].cpu().numpy() for idx in indices]

    return frames


def _resolve_label_id(row: dict[str, Any], label2id: dict[str, int] | None) -> Any:
    label = row[schema.TARGET_COLUMN]
    if label2id is not None:
        label = label2id[label]
    return label
