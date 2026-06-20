"""VideoMAE pre-processing data objects."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModel, VideoMAEImageProcessor

from aitraf_core.pre_processing.cache import (
    SampledFrameCacheContract,
    VideoMaeFeatureCacheContract,
)
from aitraf_core.processing.video import load_segmented_video_frames


@dataclass(frozen=True)
class SampledFrameSet:
    contract: SampledFrameCacheContract
    frames: list[list[torch.Tensor]]
    frame_indices: list[list[int]]


@dataclass(frozen=True)
class VideoMaeFeatureSet:
    contract: VideoMaeFeatureCacheContract
    features: torch.Tensor
    frame_indices: list[list[int]]


@dataclass(frozen=True)
class VideoMaeFeatureExtractor:
    processor: VideoMAEImageProcessor
    model: torch.nn.Module
    device: str


@lru_cache(maxsize=4)
def load_video_mae_feature_extractor(
    *,
    backbone: str,
    sample_frames: int,
    model_cache_dir: Path | str,
    device: str,
) -> VideoMaeFeatureExtractor:
    cache_dir = Path(model_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    processor = VideoMAEImageProcessor.from_pretrained(
        backbone,
        cache_dir=str(cache_dir),
    )
    backbone_config = AutoConfig.from_pretrained(
        backbone,
        cache_dir=str(cache_dir),
        trust_remote_code=True,
        num_frames=sample_frames,
    )
    model = AutoModel.from_pretrained(
        backbone,
        config=backbone_config,
        cache_dir=str(cache_dir),
        trust_remote_code=True,
    ).to(device)
    model.eval()
    return VideoMaeFeatureExtractor(
        processor=processor,
        model=model,
        device=device,
    )


def prepare_clip_pixel_values(
    *,
    video_id: str,
    processor: VideoMAEImageProcessor,
    clips_dir: Path | str,
    num_clips: int,
    sample_frames: int,
    sampling_dist: str,
) -> tuple[torch.Tensor, list[list[int]]]:
    segments, frame_indices = load_segmented_video_frames(
        video_id=video_id,
        local_clips_dir=clips_dir,
        num_segments=num_clips,
        frames_per_segment=sample_frames,
        sampling_dist=sampling_dist,
    )
    return processor(segments, return_tensors="pt")["pixel_values"], frame_indices


def extract_video_mae_batch_features(
    *,
    pixel_values: torch.Tensor,
    model: torch.nn.Module,
    device: str,
    num_clips: int,
) -> torch.Tensor:
    batch_size = pixel_values.shape[0]
    flat_pixel_values = pixel_values.reshape(
        batch_size * num_clips,
        *pixel_values.shape[2:],
    ).to(device, non_blocking=True)
    with torch.inference_mode():
        outputs = model(pixel_values=flat_pixel_values)
        features = outputs.last_hidden_state.mean(dim=1)
    return features.reshape(batch_size, num_clips, -1).half().cpu()


def extract_video_mae_clip_features(
    *,
    video_id: str,
    processor: VideoMAEImageProcessor,
    model: torch.nn.Module,
    clips_dir: Path | str,
    num_clips: int,
    sample_frames: int,
    sampling_dist: str,
    device: str,
) -> tuple[torch.Tensor, list[list[int]]]:
    pixel_values, frame_indices = prepare_clip_pixel_values(
        video_id=video_id,
        processor=processor,
        clips_dir=clips_dir,
        num_clips=num_clips,
        sample_frames=sample_frames,
        sampling_dist=sampling_dist,
    )
    features = extract_video_mae_batch_features(
        pixel_values=pixel_values.unsqueeze(0),
        model=model,
        device=device,
        num_clips=num_clips,
    )[0]
    return features, frame_indices


__all__ = [
    "extract_video_mae_batch_features",
    "extract_video_mae_clip_features",
    "load_video_mae_feature_extractor",
    "prepare_clip_pixel_values",
    "SampledFrameSet",
    "VideoMaeFeatureExtractor",
    "VideoMaeFeatureSet",
]
