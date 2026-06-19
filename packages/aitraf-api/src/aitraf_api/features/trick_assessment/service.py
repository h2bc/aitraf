"""Trick-assessment service."""

from __future__ import annotations

from aitraf_api.config import Settings
from aitraf_api.manifests import find_manifest_row_by_video_id
from aitraf_api.prediction import (
    LoadModel,
    PredictVideo,
    predict_manifest_row,
    validate_clip_exists,
)
from aitraf_api.schemas import InferenceResult


def predict_trick_assessment(
    *,
    video_id: str,
    settings: Settings,
    predict_video: PredictVideo,
    load_model: LoadModel | None = None,
) -> InferenceResult:
    row = find_manifest_row_by_video_id(
        manifest_path=settings.aqa.manifest_path,
        video_id=video_id,
    )
    validate_clip_exists(settings, str(row["video_id"]))
    return predict_manifest_row(
        ref=settings.aqa,
        row=row,
        settings=settings,
        load_model=load_model,
        predict_video=predict_video,
    )


__all__ = ["predict_trick_assessment"]
