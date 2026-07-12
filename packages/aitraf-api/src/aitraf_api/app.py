"""FastAPI application factory."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI

from aitraf_api.config import Settings, load_settings
from aitraf_api.features import router
from aitraf_api.features.demo_predictions.artifacts import download_demo_prediction_rows
from aitraf_api.features.demo_predictions.clips import prepare_public_demo_clips
from aitraf_api.features.demo_predictions.service import match_prediction_rows


def create_app(
    *,
    settings: Settings,
) -> FastAPI:
    classification_prediction_rows, aqa_prediction_rows = (
        download_demo_prediction_rows(settings)
    )
    classification_prediction_rows, aqa_prediction_rows = match_prediction_rows(
        classification_prediction_rows,
        aqa_prediction_rows,
    )
    classification_prediction_rows = prepare_public_demo_clips(
        classification_prediction_rows,
        aws_endpoint_url=settings.aws_endpoint_url,
        source_bucket=settings.aws_bucket,
        public_bucket=settings.public_asset_bucket,
    )

    app = FastAPI(title="AITRAF Demo Predictions API", version="0.1.0")
    app.state.settings = settings
    app.state.classification_prediction_rows = classification_prediction_rows
    app.state.aqa_prediction_rows = aqa_prediction_rows
    app.include_router(router)
    return app


def create_app_from_env() -> FastAPI:
    return create_app(settings=load_settings(env=os.environ, root=Path.cwd()))


__all__ = ["create_app", "create_app_from_env"]
