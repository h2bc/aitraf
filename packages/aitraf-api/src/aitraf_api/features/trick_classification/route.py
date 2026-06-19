"""Trick-classification route."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from aitraf_api.auth import require_app_token
from aitraf_api.config import Settings
from aitraf_api.dependencies import (
    get_load_model,
    get_predict_video,
    get_settings,
)
from aitraf_api.features.trick_classification.service import (
    predict_trick_classification,
)
from aitraf_api.prediction import LoadModel, PredictVideo
from aitraf_api.schemas import InferenceResult

router = APIRouter()


@router.post(
    "/inference/trick-classification/{id}",
    response_model=InferenceResult,
    dependencies=[Depends(require_app_token)],
)
def trick_classification_inference(
    id: str,
    settings: Settings = Depends(get_settings),
    load_model: LoadModel | None = Depends(get_load_model),
    predict_video: PredictVideo = Depends(get_predict_video),
) -> InferenceResult:
    if not id.strip():
        raise HTTPException(status_code=422, detail="Inference id is required")

    return predict_trick_classification(
        video_id=id,
        settings=settings,
        load_model=load_model,
        predict_video=predict_video,
    )


__all__ = ["router"]
