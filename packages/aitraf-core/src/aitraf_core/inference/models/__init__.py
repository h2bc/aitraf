"""Model-specific inference helpers."""

from aitraf_core.inference.models.video_mae import (
    predict_video_mae_label,
    predict_video_mae_logits,
)

__all__ = [
    "predict_video_mae_label",
    "predict_video_mae_logits",
]
