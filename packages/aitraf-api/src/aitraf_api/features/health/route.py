"""Health route."""

from __future__ import annotations

from fastapi import APIRouter

from aitraf_api.features.health.service import get_health
from aitraf_api.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return get_health()


__all__ = ["router"]
