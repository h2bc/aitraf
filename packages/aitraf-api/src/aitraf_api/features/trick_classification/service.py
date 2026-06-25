"""Trick-classification service."""

from __future__ import annotations

from typing import Any

from aitraf_api.config import Settings
from aitraf_api.schemas import DisplayResult, InferenceResult, ModelInfo, PredictionResult
from aitraf_api.video_loading import load_video_row


def predict_trick_classification(
    *,
    video_id: str,
    settings: Settings,
    model: Any,
) -> InferenceResult:
    row = load_video_row(
        manifest_path=settings.classification.manifest_path,
        clips_dir=settings.clips_dir,
        video_id=video_id,
    )

    label, confidence = predict_trick_classification_video_mae(
        loaded_model=model,
        video_id=video_id,
        local_clips_dir=settings.clips_dir,
    )

    return InferenceResult(
        video_id=video_id,
        prediction=PredictionResult(label=label, confidence=confidence),
        ground_truth=DisplayResult(
            label=str(row[settings.classification.ground_truth_field])
        ),
        model=ModelInfo(kind=settings.classification.model_kind),
    )


def predict_trick_classification_video_mae(**kwargs):
    from aitraf_core.inference.tasks.trick_classification.video_mae import (
        predict_trick_classification_video_mae as predict,
    )

    return predict(**kwargs)


__all__ = [
    "predict_trick_classification",
    "predict_trick_classification_video_mae",
]
