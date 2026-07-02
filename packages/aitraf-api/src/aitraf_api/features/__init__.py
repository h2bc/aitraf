"""Feature route registration."""

from fastapi import APIRouter

from aitraf_api.features.demo_predictions import router as demo_predictions_router
from aitraf_api.features.health import router as health_router

router = APIRouter()
router.include_router(health_router)
router.include_router(demo_predictions_router)

__all__ = ["router"]
