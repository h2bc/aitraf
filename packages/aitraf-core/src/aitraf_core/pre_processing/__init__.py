"""Shared video-to-model-input pre-processing helpers."""

from aitraf_core.pre_processing.cache import (
    SampledFrameCacheContract,
    VideoMaeFeatureCacheContract,
    load_cached_payload,
    save_cached_payload,
)
from aitraf_core.pre_processing.models import (
    SampledFrameSet,
    VideoMaeFeatureExtractor,
    VideoMaeFeatureSet,
    cached_video_mae_feature_extraction,
    extract_video_mae_batch_features,
    extract_video_mae_clip_features,
    load_video_mae_features,
    load_video_mae_feature_extractor,
    prepare_clip_pixel_values,
    save_video_mae_features,
    video_feature_cache_dir,
    video_feature_cache_path,
)

__all__ = [
    "load_cached_payload",
    "cached_video_mae_feature_extraction",
    "extract_video_mae_batch_features",
    "extract_video_mae_clip_features",
    "load_video_mae_features",
    "load_video_mae_feature_extractor",
    "prepare_clip_pixel_values",
    "save_video_mae_features",
    "save_cached_payload",
    "SampledFrameCacheContract",
    "SampledFrameSet",
    "VideoMaeFeatureExtractor",
    "VideoMaeFeatureCacheContract",
    "VideoMaeFeatureSet",
    "video_feature_cache_dir",
    "video_feature_cache_path",
]
