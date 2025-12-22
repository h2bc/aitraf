"""Model input processing helpers."""

from .pose_tcn import process_sample
from .video_mae import process_sample as process_video_mae_sample

__all__ = [
    "process_sample",
    "process_video_mae_sample",
]
