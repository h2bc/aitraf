"""Trick-assessment route."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from aitraf_api.auth import require_app_token
from aitraf_api.config import Settings
from aitraf_api.dependencies import (
    get_aqa_feature_extractor,
    get_aqa_model,
    get_aqa_pre_processing,
    get_settings,
)
from aitraf_api.features.trick_assessment.service import predict_trick_assessment
from aitraf_api.schemas import InferenceResult

router = APIRouter()


@router.post(
    "/inference/trick-aqa/{id}",
    response_model=InferenceResult,
    dependencies=[Depends(require_app_token)],
)
def trick_assessment_inference(
    id: str,
    cache_video_features: bool = True,
    settings: Settings = Depends(get_settings),
    loaded_model=Depends(get_aqa_model),
    feature_extractor=Depends(get_aqa_feature_extractor),
    pre_processing_config=Depends(get_aqa_pre_processing),
) -> InferenceResult:
    if not id.strip():
        raise HTTPException(status_code=422, detail="Inference id is required")

    return predict_trick_assessment(
        video_id=id,
        settings=settings,
        loaded_model=loaded_model,
        feature_extractor=feature_extractor,
        pre_processing_config=pre_processing_config,
        cache_video_features=cache_video_features,
    )


__all__ = ["router"]
