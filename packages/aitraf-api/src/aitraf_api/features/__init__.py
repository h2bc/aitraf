"""Feature route registration."""

from fastapi import APIRouter

from aitraf_api.features.demo_videos import router as demo_videos_router
from aitraf_api.features.health import router as health_router
from aitraf_api.features.trick_assessment import router as trick_assessment_router
from aitraf_api.features.trick_classification import router as trick_classification_router

router = APIRouter()
router.include_router(health_router)
router.include_router(demo_videos_router)
router.include_router(trick_classification_router)
router.include_router(trick_assessment_router)

__all__ = ["router"]
