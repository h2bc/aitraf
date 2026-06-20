"""Trick-assessment service."""

from __future__ import annotations

from typing import Any

from aitraf_core.inference import predict_temporal_fusion_label as predict
from aitraf_core.pre_processing import (
    cached_video_mae_feature_extraction as pre_processing,
    video_feature_cache_path,
)
from aitraf_core.processing.models.video_mae_temporal_fusion import (
    process_temporal_fusion_features as processing,
)
from aitraf_api.config import Settings, TrickAssessmentPreProcessingConfig
from aitraf_api.schemas import DisplayResult, InferenceResult, ModelInfo, PredictionResult
from aitraf_api.video_loading import load_video_row


def predict_trick_assessment(
    *,
    video_id: str,
    settings: Settings,
    loaded_model: Any,
    feature_extractor: Any,
    pre_processing_config: TrickAssessmentPreProcessingConfig,
    cache_video_features: bool = True,
) -> InferenceResult:
    row = load_video_row(
        manifest_path=settings.aqa.manifest_path,
        clips_dir=settings.clips_dir,
        video_id=video_id,
    )

    feature_path = video_feature_cache_path(
        feature_cache_dir=pre_processing_config.feature_cache_dir,
        video_id=video_id,
    )

    features = pre_processing(
        video_id=video_id,
        clips_dir=settings.clips_dir,
        feature_path=feature_path,
        feature_extractor=feature_extractor,
        backbone=pre_processing_config.backbone,
        num_clips=pre_processing_config.num_clips,
        sample_frames=pre_processing_config.sample_frames,
        sampling_dist=pre_processing_config.sampling_dist,
        cache_video_features=cache_video_features,
    )

    processed_features = processing(features)

    label, confidence = predict(
        model=loaded_model.model,
        features=processed_features,
        id2label=loaded_model.model.config.id2label,
    )

    return InferenceResult(
        video_id=video_id,
        prediction=PredictionResult(label=label, confidence=confidence),
        ground_truth=DisplayResult(label=str(row[settings.aqa.ground_truth_field])),
        model=ModelInfo(kind=settings.aqa.model_kind),
    )


__all__ = ["predict_trick_assessment"]
