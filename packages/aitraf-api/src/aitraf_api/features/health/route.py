"""Health route."""

from __future__ import annotations

from fastapi import APIRouter

from aitraf_api.features.health.schemas import HealthResponse
from aitraf_api.features.health.service import get_health

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return get_health()


__all__ = ["router"]
