"""VideoMAE temporal-fusion trick-assessment inference."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from aitraf_core.inference.models.video_mae_temporal_fusion import (
    predict as predict_temporal_fusion,
)
from aitraf_core.loading import HuggingFaceModel, MlflowTorchModel
from aitraf_core.pre_processing.models.video_mae_temporal_fusion import (
    predict_segment_feature_vectors_with_cache,
)


def predict_trick_assessment_video_mae_temporal_fusion(
    *,
    loaded_model: MlflowTorchModel,
    feature_extractor: HuggingFaceModel,
    video_id: str,
    clips_dir: Path,
    feature_path: Path | str,
    num_clips: int,
    sample_frames: int,
    sampling_dist: str,
    id2label: Mapping[int | str, str],
) -> tuple[str, float]:
    features = predict_segment_feature_vectors_with_cache(
        video_id=video_id,
        clips_dir=clips_dir,
        feature_path=feature_path,
        feature_extractor=feature_extractor,
        num_clips=num_clips,
        sample_frames=sample_frames,
        sampling_dist=sampling_dist,
    )

    return predict_temporal_fusion(
        model=loaded_model.model,
        features=features.float(),
        id2label=id2label,
    )


__all__ = ["predict_trick_assessment_video_mae_temporal_fusion"]
