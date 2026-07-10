"""Model-specific inference helpers."""

from aitraf_ml_core.inference.models.video_mae import (
    predict as predict_video_mae,
    predict_feature_vectors as predict_video_mae_feature_vectors,
)
from aitraf_ml_core.inference.models.video_mae_temporal_fusion import (
    predict as predict_video_mae_temporal_fusion,
)

__all__ = [
    "predict_video_mae",
    "predict_video_mae_feature_vectors",
    "predict_video_mae_temporal_fusion",
]
