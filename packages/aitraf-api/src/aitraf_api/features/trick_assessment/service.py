"""Trick-assessment service."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import HTTPException

from aitraf_api.config import Settings, TrickAssessmentPreProcessingConfig
from aitraf_api.schemas import (
    DisplayResult,
    InferenceResult,
    ModelInfo,
    PredictionResult,
)
from aitraf_api.video_loading import get_video_metadata


def predict_trick_assessment(
    *,
    video_id: str,
    settings: Settings,
    loaded_model: Any,
    feature_extractor: Any,
    pre_processing_config: TrickAssessmentPreProcessingConfig,
) -> InferenceResult:
    video_meta = get_video_metadata(
        manifest_path=settings.aqa.manifest_path,
        video_id=video_id,
    )

    if video_meta is None:
        raise HTTPException(
            status_code=404,
            detail=f"Selected video id is not in the current manifest: {video_id}",
        )

    clip_path = settings.clips_dir / video_id

    if not clip_path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Video file is unavailable: {clip_path}",
        )

    feature_path = pre_processing_config.feature_cache_dir / Path(video_id).with_suffix(
        ".pt"
    )

    label, confidence = predict_trick_assessment_video_mae_temporal_fusion(
        loaded_model=loaded_model,
        feature_extractor=feature_extractor,
        video_id=video_id,
        clips_dir=settings.clips_dir,
        feature_path=feature_path,
        num_clips=pre_processing_config.num_clips,
        sample_frames=pre_processing_config.sample_frames,
        sampling_dist=pre_processing_config.sampling_dist,
        id2label=pre_processing_config.id2label,
    )

    return InferenceResult(
        video_id=video_id,
        prediction=PredictionResult(label=label, confidence=confidence),
        ground_truth=DisplayResult(
            label=str(video_meta[settings.aqa.ground_truth_field])
        ),
        model=ModelInfo(kind=settings.aqa.model_kind),
    )


def predict_trick_assessment_video_mae_temporal_fusion(**kwargs):
    from aitraf_core.inference.tasks.trick_assessment.video_mae_temporal_fusion import (
        predict_trick_assessment_video_mae_temporal_fusion as predict,
    )

    return predict(**kwargs)


__all__ = [
    "predict_trick_assessment",
    "predict_trick_assessment_video_mae_temporal_fusion",
]
