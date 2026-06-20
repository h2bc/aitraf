"""VideoMAE temporal-fusion feature extraction helpers."""

from __future__ import annotations

from pathlib import Path

import torch

from aitraf_core.pre_processing.models.video_mae import (
    VideoMaeFeatureExtractor,
    extract_video_mae_clip_features,
)
from aitraf_core.utils.huggingface import hf_model_cache_dir_name


def video_feature_cache_dir(
    *,
    features_dir: Path | str,
    backbone: str,
    num_clips: int,
    sample_frames: int,
    sampling_dist: str,
) -> Path:
    return (
        Path(features_dir)
        / hf_model_cache_dir_name(backbone)
        / f"clips_{num_clips}_frames_{sample_frames}_sampling_{sampling_dist}"
    )


def video_feature_cache_path(
    *,
    feature_cache_dir: Path | str,
    video_id: str,
) -> Path:
    return Path(feature_cache_dir) / Path(video_id).with_suffix(".pt")


def load_video_mae_features(feature_path: Path | str) -> torch.Tensor:
    payload = torch.load(feature_path, map_location="cpu", weights_only=False)
    return torch.as_tensor(payload["features"])


def save_video_mae_features(
    *,
    feature_path: Path | str,
    video_id: str,
    backbone: str,
    num_clips: int,
    sample_frames: int,
    sampling_dist: str,
    frame_indices: list[list[int]],
    features: torch.Tensor,
) -> Path:
    feature_path = Path(feature_path)
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "video_id": video_id,
            "backbone": backbone,
            "num_clips": num_clips,
            "sample_frames": sample_frames,
            "sampling_dist": sampling_dist,
            "frame_indices": frame_indices,
            "features": features.half().cpu(),
        },
        feature_path,
    )
    return feature_path


def cached_video_mae_feature_extraction(
    *,
    video_id: str,
    clips_dir: Path | str,
    feature_path: Path | str,
    feature_extractor: VideoMaeFeatureExtractor,
    backbone: str,
    num_clips: int,
    sample_frames: int,
    sampling_dist: str,
    cache_video_features: bool,
    force: bool = False,
) -> torch.Tensor:
    feature_path = Path(feature_path)

    if cache_video_features and feature_path.exists() and not force:
        return load_video_mae_features(feature_path)

    features, frame_indices = extract_video_mae_clip_features(
        video_id=video_id,
        processor=feature_extractor.processor,
        model=feature_extractor.model,
        clips_dir=clips_dir,
        num_clips=num_clips,
        sample_frames=sample_frames,
        sampling_dist=sampling_dist,
        device=feature_extractor.device,
    )
    if cache_video_features:
        save_video_mae_features(
            feature_path=feature_path,
            video_id=video_id,
            backbone=backbone,
            num_clips=num_clips,
            sample_frames=sample_frames,
            sampling_dist=sampling_dist,
            frame_indices=frame_indices,
            features=features,
        )
    return features


__all__ = [
    "cached_video_mae_feature_extraction",
    "load_video_mae_features",
    "save_video_mae_features",
    "video_feature_cache_dir",
    "video_feature_cache_path",
]
