"""Model-specific inference helpers."""

from aitraf_core.inference.models.video_mae import (
    predict_video_mae_label,
    predict_video_mae_logits,
)
from aitraf_core.inference.models.video_mae_temporal_fusion import (
    predict_temporal_fusion_label,
    predict_temporal_fusion_logits,
)

__all__ = [
    "predict_temporal_fusion_label",
    "predict_temporal_fusion_logits",
    "predict_video_mae_label",
    "predict_video_mae_logits",
]
