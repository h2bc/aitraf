"""FastAPI application factory."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI

from aitraf_api.config import Settings, load_settings
from aitraf_api.features import router
from aitraf_api.features.demo_predictions.artifacts import download_demo_prediction_rows
from aitraf_api.features.demo_predictions.clips import prepare_public_demo_clips
from aitraf_api.features.demo_predictions.service import match_prediction_rows
from aitraf_api.features.visitor_count.service import (
    RedisVisitorCounter,
    VisitorCounter,
)


def create_app(
    *,
    settings: Settings,
    visitor_counter: VisitorCounter,
) -> FastAPI:
    classification_prediction_rows, aqa_prediction_rows = download_demo_prediction_rows(
        settings
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

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await visitor_counter.validate()
        try:
            yield
        finally:
            await visitor_counter.aclose()

    app = FastAPI(
        title="AITRAF Demo Predictions API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.visitor_counter = visitor_counter
    app.state.classification_prediction_rows = classification_prediction_rows
    app.state.aqa_prediction_rows = aqa_prediction_rows
    app.include_router(router)
    return app


def create_app_from_env() -> FastAPI:
    settings = load_settings(env=os.environ, root=Path.cwd())
    return create_app(
        settings=settings,
        visitor_counter=RedisVisitorCounter.from_url(settings.redis_url),
    )


__all__ = ["create_app", "create_app_from_env"]
