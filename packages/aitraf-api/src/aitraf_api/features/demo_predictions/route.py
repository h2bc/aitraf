"""Routes for precomputed demo predictions."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from aitraf_api.auth import require_app_token
from aitraf_api.schemas import DemoPrediction

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
    return request.app.state.demo_predictions


__all__ = ["router"]
