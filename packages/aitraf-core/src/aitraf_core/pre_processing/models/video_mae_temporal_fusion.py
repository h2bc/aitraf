"""VideoMAE temporal-fusion feature extraction helpers."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Callable
from typing import Any

import torch

from aitraf_core.inference.models import video_mae as video_mae_inference
from aitraf_core.cache import with_file_cache
from aitraf_core.loading import HuggingFaceModel
from aitraf_core.processing.models import video_mae as video_mae_processing
from aitraf_core.processing.sampling import sample_frame_indices
from aitraf_core.processing.video import decode_video_frames
from aitraf_core.utils.huggingface import hf_model_cache_dir_name
from torchcodec.decoders import VideoDecoder


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


def split_into_segments(
    *,
    video_id: str,
    clips_dir: Path,
    num_clips: int,
    sample_frames: int,
    sampling_dist: str,
) -> list[list[torch.Tensor]]:
    clip_path = clips_dir / video_id
    decoder = VideoDecoder(str(clip_path), dimension_order="NHWC")
    start, end = 0, len(decoder)
    total_frames = end - start

    if total_frames <= 0:
        raise ValueError(f"Video has no frames: {clip_path}")
    if num_clips <= 0:
        raise ValueError(f"num_clips must be greater than 0: {num_clips}")

    boundaries = torch.linspace(start, end, steps=num_clips + 1).long().tolist()

    segment_ranges = [
        (boundaries[idx], boundaries[idx + 1]) for idx in range(num_clips)
    ]

    segment_indices = [
        sample_frame_indices(
            frame_range=segment_range,
            num_frames=sample_frames,
            sampling_dist=sampling_dist,
            source=clip_path,
        )
        for segment_range in segment_ranges
    ]
    flat_indices = [
        frame_idx
        for segment_frame_indices in segment_indices
        for frame_idx in segment_frame_indices
    ]
    frames = decode_video_frames(
        decoder=decoder,
        video_path=clip_path,
        frame_indices=flat_indices,
    )

    return [
        frames[start : start + sample_frames]
        for start in range(0, len(frames), sample_frames)
    ]


def predict_segment_feature_vectors(
    *,
    video_id: str,
    clips_dir: Path,
    feature_extractor: HuggingFaceModel,
    num_clips: int,
    sample_frames: int,
    sampling_dist: str,
) -> torch.Tensor:
    inputs = process_segment_inputs(
        video_id=video_id,
        clips_dir=clips_dir,
        processor=feature_extractor.processor,
        num_clips=num_clips,
        sample_frames=sample_frames,
        sampling_dist=sampling_dist,
    )

    return predict_segment_feature_vector_batch(
        feature_extractor=feature_extractor,
        inputs=inputs.unsqueeze(0),
        num_clips=num_clips,
    )[0]


def process_segment_inputs(
    *,
    video_id: str,
    clips_dir: Path,
    processor: Any,
    num_clips: int,
    sample_frames: int,
    sampling_dist: str,
) -> torch.Tensor:
    segments = split_into_segments(
        video_id=video_id,
        clips_dir=clips_dir,
        num_clips=num_clips,
        sample_frames=sample_frames,
        sampling_dist=sampling_dist,
    )

    return video_mae_processing.process(
        inputs=segments,
        processor=processor,
    )


def predict_segment_feature_vector_batch(
    *,
    feature_extractor: HuggingFaceModel,
    inputs: torch.Tensor,
    num_clips: int,
) -> torch.Tensor:
    batch_size = inputs.shape[0]
    input_batch = inputs.reshape(
        batch_size * num_clips,
        *inputs.shape[2:],
    )

    feature_vectors = video_mae_inference.predict_feature_vectors(
        model=feature_extractor.model,
        inputs=input_batch.to(feature_extractor.device, non_blocking=True),
    )

    return feature_vectors.reshape(batch_size, num_clips, -1).half().cpu()


def load_video_mae_features(feature_path: Path | str) -> torch.Tensor:
    payload = torch.load(feature_path, map_location="cpu", weights_only=False)
    return torch.as_tensor(payload["features"])


def save_video_mae_features(
    feature_path: Path | str,
    features: torch.Tensor,
) -> Path:
    feature_path = Path(feature_path)
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"features": features.half().cpu()}, feature_path)
    return feature_path


def predict_segment_feature_vectors_with_cache(
    *,
    video_id: str,
    clips_dir: Path,
    feature_path: Path | str,
    feature_extractor: HuggingFaceModel,
    num_clips: int,
    sample_frames: int,
    sampling_dist: str,
    force: bool = False,
    on_cache_hit: Callable[[Path], None] | None = None,
    on_cache_write: Callable[[Path], None] | None = None,
) -> torch.Tensor:
    features = with_file_cache(
        path=feature_path,
        load=load_video_mae_features,
        save=save_video_mae_features,
        force=force,
        on_cache_hit=on_cache_hit,
        on_cache_write=on_cache_write,
        compute=lambda: predict_segment_feature_vectors(
            video_id=video_id,
            clips_dir=clips_dir,
            feature_extractor=feature_extractor,
            num_clips=num_clips,
            sample_frames=sample_frames,
            sampling_dist=sampling_dist,
        ),
    )
    if features is None:
        raise RuntimeError(f"Feature cache returned no features: {feature_path}")
    return features


__all__ = [
    "load_video_mae_features",
    "process_segment_inputs",
    "predict_segment_feature_vector_batch",
    "predict_segment_feature_vectors",
    "predict_segment_feature_vectors_with_cache",
    "save_video_mae_features",
    "split_into_segments",
    "video_feature_cache_dir",
]
