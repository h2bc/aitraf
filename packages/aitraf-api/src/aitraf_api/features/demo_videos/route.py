"""Demo-video route."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from aitraf_api.auth import require_app_token
from aitraf_api.config import Settings
from aitraf_api.dependencies import get_settings
from aitraf_api.features.demo_videos.service import list_demo_videos
from aitraf_api.schemas import DemoVideosResponse

router = APIRouter()


@router.get(
    "/demo-videos",
    response_model=DemoVideosResponse,
    dependencies=[Depends(require_app_token)],
)
def demo_videos(
    settings: Settings = Depends(get_settings),
) -> DemoVideosResponse:
    return DemoVideosResponse(videos=list_demo_videos(settings))


__all__ = ["router"]
