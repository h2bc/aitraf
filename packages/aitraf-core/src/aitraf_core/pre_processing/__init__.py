"""Shared video-to-model-input pre-processing helpers."""

from aitraf_core.cache import with_file_cache
from aitraf_core.pre_processing.models import (
    load_video_mae_features,
    process_segment_inputs,
    predict_segment_feature_vector_batch,
    predict_segment_feature_vectors,
    predict_segment_feature_vectors_with_cache,
    save_video_mae_features,
    video_feature_cache_dir,
)

__all__ = [
    "load_video_mae_features",
    "process_segment_inputs",
    "predict_segment_feature_vector_batch",
    "predict_segment_feature_vectors",
    "predict_segment_feature_vectors_with_cache",
    "save_video_mae_features",
    "video_feature_cache_dir",
    "with_file_cache",
]
