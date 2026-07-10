"""Pre-processing path helpers."""

from __future__ import annotations

from pathlib import Path

from aitraf_ml_core.utils.huggingface import hf_model_cache_dir_name


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


__all__ = ["video_feature_cache_dir"]
