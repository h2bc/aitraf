"""Model-specific pre-processing helpers."""

from aitraf_ml_core.pre_processing.models.video_mae_temporal_fusion import (
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
]
