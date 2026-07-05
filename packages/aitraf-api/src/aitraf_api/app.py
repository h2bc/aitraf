"""FastAPI application factory."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI

from aitraf_api.config import Settings, load_settings
from aitraf_api.features import router
from aitraf_api.features.demo_predictions.artifacts import download_demo_prediction_rows
from aitraf_api.features.demo_predictions.videos import create_video_url_presigner


def create_app(
    *,
    settings: Settings,
) -> FastAPI:
    classification_prediction_rows, aqa_prediction_rows = (
        download_demo_prediction_rows(settings)
    )
    presign_video_url = create_video_url_presigner(
        aws_endpoint_url=settings.aws_endpoint_url,
        aws_bucket=settings.aws_bucket,
    )

    app = FastAPI(title="AITRAF Demo Predictions API", version="0.1.0")
    app.state.settings = settings
    app.state.classification_prediction_rows = classification_prediction_rows
    app.state.aqa_prediction_rows = aqa_prediction_rows
    app.state.presign_video_url = presign_video_url
    app.include_router(router)
    return app


def create_app_from_env() -> FastAPI:
    return create_app(settings=load_settings(env=os.environ, root=Path.cwd()))


__all__ = ["create_app", "create_app_from_env"]
