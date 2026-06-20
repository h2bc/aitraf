"""Model-specific pre-processing helpers."""

from aitraf_core.pre_processing.models.video_mae import (
    SampledFrameSet,
    VideoMaeFeatureExtractor,
    VideoMaeFeatureSet,
    extract_video_mae_batch_features,
    extract_video_mae_clip_features,
    load_video_mae_feature_extractor,
    prepare_clip_pixel_values,
)
from aitraf_core.pre_processing.models.video_mae_temporal_fusion import (
    cached_video_mae_feature_extraction,
    load_video_mae_features,
    save_video_mae_features,
    video_feature_cache_dir,
    video_feature_cache_path,
)

__all__ = [
    "extract_video_mae_batch_features",
    "extract_video_mae_clip_features",
    "cached_video_mae_feature_extraction",
    "load_video_mae_features",
    "load_video_mae_feature_extractor",
    "prepare_clip_pixel_values",
    "save_video_mae_features",
    "SampledFrameSet",
    "VideoMaeFeatureExtractor",
    "VideoMaeFeatureSet",
    "video_feature_cache_dir",
    "video_feature_cache_path",
]
