"""Routes for precomputed demo predictions."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from aitraf_api.auth import require_app_token
from aitraf_api.features.demo_predictions.schemas import DemoPrediction
from aitraf_api.features.demo_predictions.service import build_demo_predictions_response

router = APIRouter(
    prefix="/demo-predictions",
    tags=["demo-predictions"],
)


@router.get(
    "",
    response_model=list[DemoPrediction],
    dependencies=[Depends(require_app_token)],
)
def list_demo_predictions(request: Request) -> list[DemoPrediction]:
    return build_demo_predictions_response(
        classification_rows=request.app.state.classification_prediction_rows,
        aqa_rows=request.app.state.aqa_prediction_rows,
    )


__all__ = ["router"]
